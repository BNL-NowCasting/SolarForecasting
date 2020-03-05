from matplotlib import pyplot as plt
import numpy as np

# takes a list of heights for each cloud layer in meters
# and velocities in pixels/interval (probably);
#       the method doesn't use them anyway...
def stitch_helper( image_set, heights, vels ):
	stitched_image = image_set.StitchedImage()

	imgs = list(image_set.images.values())
	rand_img = imgs[0] # arbitrarily selected image
	# rand_img = imgs.values().next()

	stitched_image.sz = rand_img.sz
	stitched_image.saz = rand_img.saz


#	print( rand_img.camera.max_theta * 180 /np.pi )
	max_tan = np.tan(rand_img.camera.max_theta)
	for l, h in enumerate(heights):
		if np.isnan(h):
			continue
		h /= 1.e3

		stitched_image.lon = image_set.min_lon - h * max_tan / image_set.deg2km / np.cos( image_set.median_lat * np.pi / 180 )
		stitched_image.lat = image_set.max_lat + h * max_tan / image_set.deg2km

		stitched_image.h = h
		stitched_image.v = vels[l]

		pixel_size = 2 * h * max_tan / rand_img.camera.nx
		stitched_image.pixel_size = pixel_size

		x_len = 2 * h * max_tan + image_set.x_cams
		y_len = 2 * h * max_tan + image_set.y_cams

		nstch_y = int(y_len//pixel_size)
		nstch_x = int(x_len//pixel_size)
		
#		print( h, nstch_x, nstch_y, x_len, y_len, stitched_image.lon, stitched_image.lat, max_tan )

		# print(pixel_size,xlen,ylen)
		rgb = np.zeros( (nstch_y,nstch_x,3), dtype=np.float32 )
		cnt = np.zeros( (nstch_y,nstch_x), dtype=np.uint8 )
		cm = np.zeros( (nstch_y,nstch_x), dtype=np.float32 )

		for img in imgs:
			start_x = int( (img.camera.lon - image_set.min_lon) * image_set.deg2km * np.cos(img.camera.lat*np.pi/180) / pixel_size )
			start_y = int( (image_set.max_lat - img.camera.lat) * image_set.deg2km / pixel_size )

			tmp = np.flip(img.rgb, axis=1)
			# tmp[img.cm != l+1,:] = 0				      
			mask = tmp[...,0] > 0
			# print(img.camID,ilayer,h[ilayer],start_x,start_y,mask.shape,stitched.shape)
			rgb[start_y:start_y+img.ny,start_x:start_x+img.nx][mask] += tmp[mask]
			cnt[start_y:start_y+img.ny,start_x:start_x+img.nx] += mask

			if (img.cm is not None):
				tmp = np.flip(img.cm, axis=1)
			     # tmp[img.cm != l+1,:] = 0
				cm[start_y:start_y+img.ny,start_x:start_x+img.nx][mask] += tmp[mask]

		for i in range(3):
			rgb[...,i] /= cnt
		cm /= cnt

		stitched_image.rgb = rgb.astype(np.uint8)
		stitched_image.cm = np.rint(cm).astype(np.uint8)
		image_set.stitched_image = stitched_image
		break

