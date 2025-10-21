pub fn hsva_to_rgba(hsva: Vec<f32>) -> Vec<f32> {
    assert!(hsva.len() == 4);
    let (h, s, v, a) = (hsva[0], hsva[1], hsva[2], hsva[3]);

    if s == 0.0 {
        // Achromatic (gray)
        return vec![v, v, v, a];
    }

    let h = (h % 1.0) * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    let (r, g, b) = match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };

    vec![r, g, b, a]
}
