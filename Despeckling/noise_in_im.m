im=imread('test_image.jpg');
%imshow(im);

%%%%%%%�����˹������ע���ֵ�ͷ���ĵ�λ����0����ȫ�ڣ�1����ȫ��
im_n=imnoise(im,'gaussian',-0.5,0.1);
imshow(im_n);
