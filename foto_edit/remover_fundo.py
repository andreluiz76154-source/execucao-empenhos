import cv2, numpy as np
src='foto_final_4x5_com_fundo.jpg'; dst='foto_final_4x5.jpg'
im=cv2.imread(src); ox,oy=600,450   # offsets of the inspected crop
m=np.zeros(im.shape[:2],np.uint8)
def ell(cx,cy,ax,ay,ang=0): cv2.ellipse(m,(ox+cx,oy+cy),(ax,ay),ang,0,360,255,-1)
ell(230,225,38,82,5)    # runner 1 (legs+body)
ell(292,205,42,80,5)    # runner 2
ell(310,150,30,25)      # head/shoulders area of runner 2 on the grass
m=cv2.dilate(m,np.ones((9,9),np.uint8))
# third partial figure behind his neck: thin strip, kept away from the face
p=np.zeros_like(m); cv2.ellipse(p,(ox+98,oy+345),(26,100),0,0,360,255,-1)
m=cv2.bitwise_or(m,cv2.dilate(p,np.ones((7,7),np.uint8)))
cv2.imwrite('/tmp/mask.png',m)
fill=cv2.inpaint(im,m,12,cv2.INPAINT_TELEA)
# soften inpainted area to match the blurred background, then re-add grain
soft=cv2.GaussianBlur(fill,(0,0),4)
a=cv2.GaussianBlur(m,(0,0),5).astype(np.float32)[...,None]/255.
rng=np.random.default_rng(3)
gr=cv2.GaussianBlur(rng.normal(0,1,m.shape).astype(np.float32),(0,0),0.7)*2.2
base=fill.astype(np.float32)*0.4+soft.astype(np.float32)*0.6+gr[...,None]
out=np.clip(im.astype(np.float32)*(1-a)+base*a,0,255).astype(np.uint8)
cv2.imwrite(dst,out,[cv2.IMWRITE_JPEG_QUALITY,95])
cv2.imwrite('/tmp/bg_after.png',out[450:1150,600:1100])

# ---- pass 2: clean residual smudges ----
im=cv2.imread(dst).astype(np.float32); ox,oy=600,450
# (a) dark smear on the path: clone clean path texture from just below, feathered
m2=np.zeros(m.shape,np.uint8); cv2.ellipse(m2,(ox+255,oy+262),(48,58),0,0,360,255,-1)
a2=cv2.GaussianBlur(m2,(0,0),14).astype(np.float32)[...,None]/255.
shifted=np.roll(im,-85,axis=0)   # content from 85px below
im=im*(1-a2)+shifted*a2
# (b) grey-green blotch on the grass: fill from strongly blurred surroundings
m3=np.zeros(m.shape,np.uint8); cv2.ellipse(m3,(ox+262,oy+168),(46,40),0,0,360,255,-1)
tmp=cv2.inpaint(im.astype(np.uint8),m3,25,cv2.INPAINT_NS)
tmp=cv2.GaussianBlur(tmp,(0,0),6).astype(np.float32)
a3=cv2.GaussianBlur(m3,(0,0),12).astype(np.float32)[...,None]/255.
im=im*(1-a3)+tmp*a3
out=np.clip(im+gr[...,None]*(a2+a3),0,255).astype(np.uint8)
cv2.imwrite(dst,out,[cv2.IMWRITE_JPEG_QUALITY,95])
cv2.imwrite('/tmp/bg_after.png',out[450:1150,600:1100])
cv2.imwrite('/tmp/full_small.jpg',cv2.resize(out,(720,900)))
