fn _max_pixel_diff(
    lhs: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    rhs: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
) -> u8 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap()
}

#[test]
fn render_gray() -> Result<(), Box<dyn std::error::Error>> {
    let mut im = mathmap::exec_mathmap_script("examples/render/gray.mm", 256, 256, 1)?;
    let im_ref = image::open("tests/render_gray.png")?;
    assert_eq!(im.next().unwrap(), im_ref.into());
    Ok(())
}

#[test]
fn render_grid() -> Result<(), Box<dyn std::error::Error>> {
    let mut im = mathmap::exec_mathmap_script("examples/render/grid.mm", 256, 256, 1)?;
    let im_ref = image::open("tests/render_grid.png")?;
    assert_eq!(im.next().unwrap(), im_ref.into());
    Ok(())
}

#[test]
#[ignore]
fn render_moire1() -> Result<(), Box<dyn std::error::Error>> {
    let mut im = mathmap::exec_mathmap_script("examples/render/moire_1.mm", 256, 256, 1)?;
    let im_ref = image::open("tests/render_moire_1.png")?.to_rgba8();
    assert_eq!(im.next().unwrap(), im_ref.into());
    Ok(())
}

#[test]
fn render_moire2() -> Result<(), Box<dyn std::error::Error>> {
    let mut im = mathmap::exec_mathmap_script("examples/render/moire_2.mm", 256, 256, 1)?;
    let im_ref = image::open("tests/render_moire_2.png")?.to_rgba8();
    assert_eq!(im.next().unwrap(), im_ref.into());
    Ok(())
}

#[test]
fn render_spiral() -> Result<(), Box<dyn std::error::Error>> {
    let mut im = mathmap::exec_mathmap_script("examples/render/spiral.mm", 256, 256, 1)?;
    let im_ref = image::open("tests/render_spiral.png")?.to_rgba8();
    assert_eq!(im.next().unwrap(), im_ref.into());
    Ok(())
}
