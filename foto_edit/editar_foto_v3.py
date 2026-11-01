import cv2, numpy as np
from PIL import Image
src='/root/.claude/uploads/1cbe492a-d1a3-5736-84e7-bfa68eb4bde6/afddd1b4-image.jpg'
a=cv2.cvtColor(np.array(Image.open(src).convert('RGB')),cv2.COLOR_RGB2BGR)
y0=100; c=a[y0:y0+1125].copy()                      # 900x1125 (4:5)

# 1) kill blotchy compression colour: strong chroma denoise, light luma denoise
lab=cv2.cvtColor(c,cv2.COLOR_BGR2LAB)
L,A,B=cv2.split(lab)
A=cv2.bilateralFilter(A,9,12,9); B=cv2.bilateralFilter(B,9,12,9)
A=cv2.GaussianBlur(A,(0,0),1.5); B=cv2.GaussianBlur(B,(0,0),1.5)
L=cv2.fastNlMeansDenoising(L,None,3,5,15)
# 2) remove background people at native scale (coords from the 2x work /2)
m=np.zeros(L.shape,np.uint8)
for cx,cy,ax,ay in [(230,225,38,82),(292,205,42,80),(310,150,30,25)]:
    cv2.ellipse(m,(300+(cx)//2,225+(cy)//2),(ax//2,ay//2),5,0,360,255,-1)
cv2.ellipse(m,(300+98//2,225+345//2),(13,50),0,0,360,255,-1)
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
out=cv2.resize(out,(1080,1350),interpolation=cv2.INTER_LANCZOS4)
Y=cv2.cvtColor(out,cv2.COLOR_BGR2YCrCb).astype(np.float32)
bl=cv2.GaussianBlur(Y[...,0],(0,0),1.3); d=Y[...,0]-bl
d=np.where(np.abs(d)<1.5,0,d)
lo=cv2.erode(Y[...,0],np.ones((5,5)));hi=cv2.dilate(Y[...,0],np.ones((5,5)))
Y[...,0]=np.clip(Y[...,0]+0.6*d,lo-5,hi+5)
out=cv2.cvtColor(np.clip(Y,0,255).astype(np.uint8),cv2.COLOR_YCrCb2BGR)
rng=np.random.default_rng(2)
g=cv2.GaussianBlur(rng.normal(0,1,out.shape[:2]).astype(np.float32),(0,0),0.6)*1.6
out=np.clip(out.astype(np.float32)+g[...,None],0,255).astype(np.uint8)
im=Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
im.save('foto_instagram_1080x1350.jpg',quality=95,subsampling=0,optimize=True)
im.crop((250,100,1080,700)).save('/tmp/v3_face.jpg',quality=92)
im.resize((720,900)).save('/tmp/v3_small.jpg',quality=92)
