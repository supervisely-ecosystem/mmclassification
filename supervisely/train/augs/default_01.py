import imgaug.augmenters as iaa

seq = iaa.Sequential([
	iaa.Sometimes(0.5, iaa.blur.GaussianBlur(sigma=(0, 3))),
	iaa.Sometimes(0.5, iaa.arithmetic.AdditiveGaussianNoise(scale=(0, 15), per_channel=False)),
	iaa.Sometimes(0.25, iaa.imgcorruptlike.MotionBlur(severity=(1, 5))),
	iaa.Sometimes(0.5, iaa.geometric.Rotate(rotate=(-30, 30), order=1, cval=0, mode='reflect', fit_output=False))
], random_order=True)
