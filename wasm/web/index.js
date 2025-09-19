import init, { make_image, compile_to_wgsl } from "./pkg/mathmap_wasm.js";

function base64EncodeUtf8(input) {
    const bytes = new TextEncoder().encode(input);
    let binary = "";
    for (const byte of bytes) {
        binary += String.fromCharCode(byte);
    }
    return btoa(binary);
}

function base64DecodeUtf8(encoded) {
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    return new TextDecoder().decode(bytes);
}

function readStateFromFragment() {
    const hash = window.location.hash.startsWith("#")
        ? window.location.hash.slice(1)
        : window.location.hash;
    if (!hash) {
        return {};
    }
    const params = new URLSearchParams(hash);
    const state = {};
    const scriptParam = params.get("script");
    if (scriptParam) {
        try {
            state.script = base64DecodeUtf8(scriptParam);
        } catch (e) {
            console.warn("Failed to decode script from URL fragment", e);
        }
    }
    const gpuParam = params.get("gpu");
    if (gpuParam === "1") {
        state.gpu = true;
    } else if (gpuParam === "0") {
        state.gpu = false;
    }
    return state;
}

let lastFragment = window.location.hash || "";

function writeStateToFragment({ script, gpu }) {
    const params = new URLSearchParams();
    if (typeof gpu === "boolean") {
        params.set("gpu", gpu ? "1" : "0");
    }
    if (typeof script === "string") {
        try {
            params.set("script", base64EncodeUtf8(script));
        } catch (e) {
            console.warn("Failed to encode script for URL fragment", e);
        }
    }
    const newHash = params.toString();
    const fragment = newHash ? `#${newHash}` : "";
    if (fragment === lastFragment) {
        return;
    }
    lastFragment = fragment;
    const newUrl = `${window.location.pathname}${window.location.search}${fragment}`;
    history.replaceState(null, "", newUrl);
}

async function run() {
    await init();
    const canvas = document.getElementById("canvas");
    const downloadBtn = document.getElementById("downloadBtn");
    const width = 256, height = 256;
    canvas.width = width;
    canvas.height = height;
    let gpuState = null;

    function triggerDownload(url, filename) {
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = filename;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
    }

    downloadBtn.addEventListener("click", () => {
        downloadBtn.disabled = true;
        const restore = () => {
            downloadBtn.disabled = false;
        };
        canvas.toBlob((blob) => {
            if (!blob) {
                restore();
                return;
            }
            const url = URL.createObjectURL(blob);
            triggerDownload(url, "mathmap.png");
            URL.revokeObjectURL(url);
            restore();
        }, "image/png");
        return;
    });

    async function ensureWebGPU() {
        if (gpuState) {
            return gpuState;
        }
        if (!navigator.gpu) {
            throw new Error("WebGPU is not supported in this browser");
        }
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error("Failed to acquire WebGPU adapter");
        }
        const device = await adapter.requestDevice();
        const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
        const context = canvas.getContext("webgpu");
        context.configure({ device, format: canvasFormat });
        gpuState = { device, context, canvasFormat };
        return gpuState;
    }

    async function renderWebGPU(source) {
        const { device, context, canvasFormat } = await ensureWebGPU();
        const shader_code = compile_to_wgsl(source);
        console.log(shader_code);
        const shaderModule = device.createShaderModule({ code: shader_code });

        const widthU32 = width;
        const heightU32 = height;
        const pixelCount = widthU32 * heightU32;
        const bufferSize = pixelCount * 4 * Float32Array.BYTES_PER_ELEMENT;

        const storageBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const uniformBuffer = device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.UNIFORM,
            mappedAtCreation: true,
        });
        new Uint32Array(uniformBuffer.getMappedRange()).set([widthU32, heightU32]);
        uniformBuffer.unmap();

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" },
                },
            ],
        });

        const pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: {
                module: shaderModule,
                entryPoint: "main",
            },
        });

        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: storageBuffer } },
                { binding: 1, resource: { buffer: uniformBuffer } },
            ],
        });

        const texture = device.createTexture({
            size: [width, height],
            format: "rgba32float",
            usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
        });

        const sampler = device.createSampler({
            magFilter: "nearest",
            minFilter: "nearest",
        });

        const quadShader = device.createShaderModule({
            code: `
struct VertexOutput {
    @builtin(position) pos : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx : u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>( 3.0,  1.0),
        vec2<f32>(-1.0,  1.0)
    );
    var uvCoords = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 2.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 0.0)
    );

    var out : VertexOutput;
    out.pos = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv  = uvCoords[idx];
    return out;
}

@group(0) @binding(0) var tex : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = textureSample(tex, samp, in.uv);
    return clamp(c, vec4<f32>(0.0), vec4<f32>(1.0));
}
        `,
        });

        const renderBindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: {
                        sampleType: "unfilterable-float",
                    },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: {
                        type: "non-filtering",
                    },
                },
            ],
        });

        const renderPipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [renderBindGroupLayout],
        });

        const renderPipeline = device.createRenderPipeline({
            layout: renderPipelineLayout,
            vertex: { module: quadShader, entryPoint: "vs_main" },
            fragment: {
                module: quadShader,
                entryPoint: "fs_main",
                targets: [{ format: canvasFormat }],
            },
            primitive: { topology: "triangle-list" },
        });

        const renderBindGroup = device.createBindGroup({
            layout: renderBindGroupLayout,
            entries: [
                { binding: 0, resource: texture.createView() },
                { binding: 1, resource: sampler },
            ],
        });

        const commandEncoder = device.createCommandEncoder();

        {
            // Compute pass to render the filter.
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(pipeline);
            passEncoder.setBindGroup(0, bindGroup);
            const workgroupSize = 8;
            const workgroupCountX = Math.ceil(width / workgroupSize);
            const workgroupCountY = Math.ceil(height / workgroupSize);
            passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
            passEncoder.end();
        }

        // Copy output of the compute shader to a texture that can be read
        // by the render shaders.
        commandEncoder.copyBufferToTexture(
            {
                buffer: storageBuffer,
                bytesPerRow: width * 16, // 4 floats × 4 bytes
            },
            { texture },
            { width, height }
        );

        {
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: context.getCurrentTexture().createView(),
                        loadOp: "clear",
                        clearValue: { r: 0, g: 0, b: 0, a: 1 },
                        storeOp: "store",
                    },
                ],
            });
            renderPass.setPipeline(renderPipeline);
            renderPass.setBindGroup(0, renderBindGroup);
            renderPass.draw(6);
            renderPass.end();
        }

        device.queue.submit([commandEncoder.finish()]);

        storageBuffer.destroy();
        uniformBuffer.destroy();
    }

    function setError(message) {
        const box = document.getElementById("errorBox");
        if (!message) {
            box.style.display = "none";
            box.textContent = "";
            return;
        }
        box.textContent = message;
        box.style.display = "block";
    }

    function extractMessage(e) {
        if (!e) return "Unknown error";
        if (e instanceof Error) return e.message || String(e);
        if (typeof e === "string") return e;
        try {
            return JSON.stringify(e);
        } catch (_) {
            return String(e);
        }
    }

    async function render() {
        const text = inputBox.value;
        try {
            if (gpuToggle.checked) {
                await renderWebGPU(text);
            } else {
                const data = make_image(text);
                const imgData = new ImageData(
                    new Uint8ClampedArray(data),
                    width,
                    height
                );
                const ctx = canvas.getContext("2d");
                ctx.putImageData(imgData, 0, 0);
            }
            setError(null);
        } catch (e) {
            console.error(e);
            setError(extractMessage(e));
        }
    }

    const filterTarget = `\
# This is an example Mathmap script.
# Try changing some of the numbers below
# and press "Render" to see what happens
# to the image on the right.

filter target ()
    if r % 0.4 < 0.2 then
        rgbColor(1, 0, 0)
    else
        rgbColor(1, 1, 1)
    end
end
`;

    const filterRed = `\
filter red ()
    rgbColor(0.0, 1.0, 0.0)
end
`;
    const isLocalhost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
    const fragmentState = readStateFromFragment();
    const inputBox = document.getElementById("inputBox");
    const gpuToggle = document.getElementById("gpuToggle");

    inputBox.value = fragmentState.script ?? (isLocalhost ? filterRed : filterTarget);

    gpuToggle.checked = fragmentState.gpu ?? isLocalhost;

    writeStateToFragment({ script: inputBox.value, gpu: gpuToggle.checked });

    void render();

    document.getElementById("renderBtn").addEventListener("click", () => {
        writeStateToFragment({ script: inputBox.value, gpu: gpuToggle.checked });
        void render();
    });

    inputBox.addEventListener("keydown", (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
            e.preventDefault();
            writeStateToFragment({ script: inputBox.value, gpu: gpuToggle.checked });
            void render();
        }
    });

    inputBox.addEventListener("input", () => {
        writeStateToFragment({ script: inputBox.value, gpu: gpuToggle.checked });
    });

    gpuToggle.addEventListener("change", () => {
        writeStateToFragment({ script: inputBox.value, gpu: gpuToggle.checked });
        // Can't use WebGPU and 2d contextes at the same time.
        window.location.reload();
    });
}

run();
