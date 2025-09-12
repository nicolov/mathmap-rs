filter render_mandelbrot ()
	pj=0;
	pk=0;
	c1=0;
	ci=0;
	cj=0;
	ck=0;
	num_iterations=32;

    pc=ri:xy;
    p=quat:[pc[0],pc[1],pj,pk];
    c=quat:[c1,ci,cj,ck];
    iter=0;
    while abs(c)<2 && iter<(num_iterations-1)
    do
        c=c*c+p;
        iter=iter+1
    end;
    grayColor(iter/num_iterations)
end
