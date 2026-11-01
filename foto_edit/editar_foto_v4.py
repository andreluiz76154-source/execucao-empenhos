import cv2, numpy as np
from PIL import Image
src='/root/.claude/uploads/1cbe492a-d1a3-5736-84e7-bfa68eb4bde6/afddd1b4-image.jpg'
a=cv2.cvtColor(np.array(Image.open(src).convert('RGB')),cv2.COLOR_RGB2BGR)
c=a.copy()                                          # full frame 900x1600 (9:16)

# 1) kill blotchy compression colour: strong chroma denoise, light luma denoise
lab=cv2.cvtColor(c,cv2.COLOR_BGR2LAB)
L,A,B=cv2.split(lab)
A=cv2.bilateralFilter(A,9,12,9); B=cv2.bilateralFilter(B,9,12,9)
A=cv2.GaussianBlur(A,(0,0),1.5); B=cv2.GaussianBlur(B,(0,0),1.5)
L=cv2.fastNlMeansDenoising(L,None,3,5,15)
# 2) remove background people at native scale (coords from the 2x work /2)
m=np.zeros(L.shape,np.uint8)
for cx,cy,ax,ay in [(230,225,38,82),(292,205,42,80),(310,150,30,25)]:
    cv2.ellipse(m,(300+(cx)//2,325+(cy)//2),(ax//2,ay//2),5,0,360,255,-1)
cv2.ellipse(m,(300+98//2,325+345//2),(13,50),0,0,360,255,-1)
m=cv2.dilate(m,np.ones((5,5),np.uint8))
cl=cv2.merge([L,A,B]); bgr=cv2.cvtColor(cl,cv2.COLOR_LAB2BGR)
fill=cv2.inpaint(bgr,m,8,cv2.INPAINT_TELEA)
fill=cv2.GaussianBlur(fill,(0,0),2.5)
al=cv2.GaussianBlur(m,(0,0),3).astype(np.float32)[...,None]/255.
bgr=np.clip(bgr*(1-al)+fill*al,0,255).astype(np.uint8)
# 3) tone: moderate shadow lift, gentle contrast, keep deep blacks
lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB).astype(np.float32)
Lf=lab[...,0]/255.
L2=Lf**0.80
L2=0.8*L2+0.2*(L2*L2*(3-2*L2))
L2=np.where(Lf<0.05,Lf+(L2-Lf)*(Lf/0.05),L2)
lab[...,0]=np.clip(L2*255,0,255)
lab[...,1:]=128+(lab[...,1:]-128)*1.08
out=cv2.cvtColor(lab.astype(np.uint8),cv2.COLOR_LAB2BGR)
# 4) deliver at Instagram 4:5 size 1080x1350, then sharpen once at final size
out=cv2.resize(out,(1080,1920),interpolation=cv2.INTER_LANCZOS4)
Y=cv2.cvtColor(out,cv2.COLOR_BGR2YCrCb).astype(np.float32)
y=Y[...,0]/255.
# Richardson-Lucy deconvolution (gaussian PSF) to recover real edge detail
k=cv2.getGaussianKernel(9,1.3); psf=k@k.T
est=y.copy()
for _ in range(10):
    conv=cv2.filter2D(est,-1,psf,borderType=cv2.BORDER_REFLECT)
    est=est*cv2.filter2D(y/np.maximum(conv,1e-4),-1,psf[::-1,::-1],borderType=cv2.BORDER_REFLECT)
    est=np.clip(est,0,1)
y2=0.65*est+0.35*y            # blend to keep it natural / limit noise
y2=y2*255
# fine + medium unsharp masks, with noise threshold and overshoot clamp (no halos)
d1=y2-cv2.GaussianBlur(y2,(0,0),1.0); d1=np.where(np.abs(d1)<1.5,0,d1)
d2=y2-cv2.GaussianBlur(y2,(0,0),3.0)
y3=y2+0.55*d1+0.22*d2
lo=cv2.erode(y2,np.ones((5,5)));hi=cv2.dilate(y2,np.ones((5,5)))
Y[...,0]=np.clip(y3,lo-4,hi+4)
out=cv2.cvtColor(np.clip(Y,0,255).astype(np.uint8),cv2.COLOR_YCrCb2BGR)
rng=np.random.default_rng(2)
g=cv2.GaussianBlur(rng.normal(0,1,out.shape[:2]).astype(np.float32),(0,0),0.6)*1.6
out=np.clip(out.astype(np.float32)+g[...,None],0,255).astype(np.uint8)
im=Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
im.save('foto_fullhd_1080x1920.jpg',quality=95,subsampling=0,optimize=True)
im.crop((300,250,1080,850)).save('/tmp/v3_face.jpg',quality=92)
im.resize((540,960)).save('/tmp/v3_small.jpg',quality=92)
