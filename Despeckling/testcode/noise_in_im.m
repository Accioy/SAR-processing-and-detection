im=imread('test_image.jpg');
%imshow(im);

%%%%%%%加入高斯噪声，注意均值和方差的单位，以0代表全黑，1代表全白
im_n=imnoise(im,'gaussian',-0.5,0.1);
imshow(im_n);
