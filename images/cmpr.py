from PIL import Image

foo = Image.open('panda.jpg')  # My image is a 200x374 jpeg that is 102kb large
foo.size  # (200, 374)
 
# downsize the image with an ANTIALIAS filter (gives the highest quality)
foo = foo.resize((1020,510),Image.Resampling.LANCZOS)
 
#foo.save('bird_2_cmpr.jpg', quality=95)  # The saved downsized image size is 24.8kb
 
foo.save('cmpr.jpg', optimize=True, quality=95)  # The saved downsized image size is 22.9kb