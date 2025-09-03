filter disco ()
	red_wavelength=10;
	green_wavelength=15;
	blue_wavelength=20;
	zoom=500;
    rl=red_wavelength;
    gl=green_wavelength;
    bl=blue_wavelength;
    q=t*2*pi;
    rz=r*zoom;
    abs(rgba:[sin(rz/rl+q)+sin(a*rl+q),
              sin(rz/gl+q)+sin(a*gl+q),
              sin(rz/bl+q)+sin(a*bl+q),2])
end
