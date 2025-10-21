filter plasma ()
    x = xy[0]*5;
    y = xy[1]*5;

    v = sin(x) + sin(y) + sin(x+y) + sin(sqrt(x*x + y*y));
    hsva:[(v + 4) / 8, 0.9, 1, 1]
end
