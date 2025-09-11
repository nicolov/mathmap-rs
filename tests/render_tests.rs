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
    let im = mathmap::exec_mathmap_script("examples/render/moire_1.mm", 256, 256, 1)?
        .next()
        .unwrap();
    let im_ref = image::open("tests/render_moire_1.png")?.to_rgba8();
    // assert_eq!(im.next().unwrap(), im_ref.into());
    let max_diff = im
        .iter()
        .zip(im_ref.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    println!("max diff: {}", max_diff);
    Ok(())
}

#[test]
#[ignore]
fn render_spiral() -> Result<(), Box<dyn std::error::Error>> {
    let im = mathmap::exec_mathmap_script("examples/render/spiral.mm", 256, 256, 1)?
        .next()
        .unwrap();
    let im_ref = image::open("tests/render_spiral.png")?.to_rgba8();
    let max_diff = im
        .iter()
        .zip(im_ref.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    println!("max diff: {}", max_diff);
    Ok(())
}
