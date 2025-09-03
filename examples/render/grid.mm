filter grid ()
	width=0.2;
	height=0.2;
	thickness=0.02;
  nxy = abs(xy) + thickness/2;
  grayColor(if (nxy[0]%width)<=thickness || (nxy[1]%height)<=thickness then 0 else 1 end)
end
