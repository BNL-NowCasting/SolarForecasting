These systemd service config files `image_downloader_{alb,bnl}.service` should be
installed into the directory returned by:

`pkg-config systemd --variable=systemdsystemunitdir`
	
It runs the commands in `/home/nowcast/run/image_downloader_{alb,bnl}.sh`.
To enable add boot

`systemctl enable image_downloader_{alb,bnl}`
	
To start

`systemctl start image_downloader_{alb,bnl}`
	
The status is shown with

`systemctl status -l image_downloader_{alb,bnl}`
	
Note: This also shows the subprocesses started, and the last error
messages.

The process is supposed to restart with a 1 s delay when it dies and
disable itself if there are more than 5 restarts in a 10 s interval.
Killing a sub-process restarts automatically within 1 s.
