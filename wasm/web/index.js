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
    const ctx = canvas.getContext("2d");
    const width = 256, height = 256;
    canvas.width = width;
    canvas.height = height;
    let gpuState = null;

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

    function getErrorSink(renderPass) {
        let log = "";
        const flush = () => {
            if (log) {
                console.error("WGSL validation failed:\n" + log);
                log = "";
            }
        };
        renderPass.addEventListener("error", (event) => {
            log += `${event.message}\n`;
        });
        renderPass.addEventListener("validationcomplete", flush);
        return { flush };
    }

    async function renderWebGPU(source) {
        const { device, context, canvasFormat } = await ensureWebGPU();
        const renderPass = device.createCommandEncoder();
        const { shader, errors } = compileToWgsl(source);
        if (errors) {
            throw new Error(errors);
        }
        const shaderModule = device.createShaderModule({ code: shader });
        const errorSink = getErrorSink(shaderModule);
        errorSink.flush();

        const widthU32 = width;
        const heightU32 = height;
        const pixelCount = widthU32 * heightU32;
        const bufferSize = pixelCount * 4 * Float32Array.BYTES_PER_ELEMENT;

        const storageBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const stagingBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
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
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {} },
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

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        const workgroupSize = 8;
        const workgroupCountX = Math.ceil(width / workgroupSize);
        const workgroupCountY = Math.ceil(height / workgroupSize);
        passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
        passEncoder.end();

        commandEncoder.copyBufferToBuffer(storageBuffer, 0, stagingBuffer, 0, bufferSize);

        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const data = stagingBuffer.getMappedRange();
        const floatPixels = new Float32Array(data);
        const rgba = new Uint8ClampedArray(pixelCount * 4);
        for (let i = 0; i < pixelCount; i++) {
            const base = i * 4;
            rgba[base + 0] = Math.min(255, Math.max(0, Math.round(floatPixels[base + 0] * 255)));
            rgba[base + 1] = Math.min(255, Math.max(0, Math.round(floatPixels[base + 1] * 255)));
            rgba[base + 2] = Math.min(255, Math.max(0, Math.round(floatPixels[base + 2] * 255)));
            rgba[base + 3] = Math.min(255, Math.max(0, Math.round(floatPixels[base + 3] * 255)));
        }
        stagingBuffer.unmap();
        storageBuffer.destroy();
        stagingBuffer.destroy();
        uniformBuffer.destroy();
        const imgData = new ImageData(rgba, width, height);
        ctx.putImageData(imgData, 0, 0);
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
        void render();
    });
}

run();
