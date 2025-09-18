filter heart ()
    rR = 218/255;  rG = 41/255;  rB = 28/255;

    red = rgbColor(rR, rG, rB);
    white = rgbColor(1, 1, 1);

    X = x * 1.4;
    Y = (y + 0.15) * 1.4;

    f = (X*X + Y*Y - 1)^3 - (X*X)*(Y*Y*Y);

    if f <= 0 then
        white
    else
        red
    end;
end
