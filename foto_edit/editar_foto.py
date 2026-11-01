import cv2, numpy as np
from PIL import Image, ImageOps
src='/root/.claude/uploads/1cbe492a-d1a3-5736-84e7-bfa68eb4bde6/afddd1b4-image.jpg'
out='/home/user/execucao-empenhos/foto_edit/'
orig=ImageOps.exif_transpose(Image.open(src)).convert('RGB')
a=np.array(orig)
# 1) crop 4:5 (orig coords), keeps both faces + shirts
y0=100; h=1125; crop=a[y0:y0+h]
# 2) denoise (light, luma+chroma) on the dark file before lifting it
bgr=cv2.cvtColor(crop,cv2.COLOR_RGB2BGR)
dn=cv2.fastNlMeansDenoisingColored(bgr,None,2,3,7,21)
# 3) exposure: lift shadows via curve on L (LAB), keep highlights
lab=cv2.cvtColor(dn,cv2.COLOR_BGR2LAB).astype(np.float32)
L=lab[...,0]/255.
gamma=0.84
L2=L**gamma
# gentle S-curve for contrast, blended
s=L2*L2*(3-2*L2)
L2=0.6*L2+0.4*s
# protect deep blacks: don't lift pure black
L2=np.where(L<0.04, L*(L2.max() and 1)+ (L2-L)*(L/0.04), L2)
lab[...,0]=np.clip(L2*255,0,255)
# slight chroma boost compensates lifted shadows' desaturation
lab[...,1:]=128+(lab[...,1:]-128)*1.10
# local face lift: soft skin mask on denoised crop, +~9% luminance, feathered
yc=cv2.cvtColor(dn,cv2.COLOR_BGR2YCrCb)
sk=((yc[...,1]>135)&(yc[...,1]<180)&(yc[...,2]>85)&(yc[...,2]<135)&(yc[...,0]>35)).astype(np.float32)
sk=cv2.morphologyEx(sk,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
sk=cv2.GaussianBlur(sk,(0,0),14)
Lc=lab[...,0]
lab[...,0]=np.clip(Lc+sk*(255-Lc)*0.16,0,255)
# local microcontrast (very gentle CLAHE blend)
cl=cv2.createCLAHE(clipLimit=1.6,tileGridSize=(6,6)).apply(lab[...,0].astype(np.uint8)).astype(np.float32)
lab[...,0]=lab[...,0]*0.7+cl*0.3
img=cv2.cvtColor(lab.astype(np.uint8),cv2.COLOR_LAB2BGR).astype(np.float32)
# 4) white balance: partial gray-world on mid-tones (30% strength, keeps night warmth)
m=img.reshape(-1,3); sel=(m.mean(1)>40)&(m.mean(1)<200)
avg=m[sel].mean(0); gain=(avg.mean()/avg)
gain=1+(gain-1)*0.15
img=np.clip(img*gain,0,255)
print('WB gain BGR',gain)
img=img.astype(np.uint8)
# 5) upscale 2x Lanczos
big=cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_LANCZOS4)
# 6) skin smoothing (light, texture preserved): mask on skin tones, blend 30% of bilateral
ycc=cv2.cvtColor(big,cv2.COLOR_BGR2YCrCb)
skin=((ycc[...,1]>135)&(ycc[...,1]<180)&(ycc[...,2]>85)&(ycc[...,2]<135)&(ycc[...,0]>50)).astype(np.float32)
skin=cv2.morphologyEx(skin,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
skin=cv2.GaussianBlur(skin,(0,0),8)
sm=cv2.bilateralFilter(big,9,25,9)
# keep high-frequency texture: smooth only low-mid frequencies
hf=big.astype(np.float32)-cv2.GaussianBlur(big,(0,0),2).astype(np.float32)
smooth=sm.astype(np.float32)*0.5+big.astype(np.float32)*0.5
smooth=cv2.GaussianBlur(smooth,(0,0),0.8)+hf*0.9
k=(skin*0.15)[...,None]
big=np.clip(big*(1-k)+smooth*k,0,255).astype(np.uint8)
# 7) unsharp mask, moderate, luminance only, threshold avoids amplifying noise
ycc=cv2.cvtColor(big,cv2.COLOR_BGR2YCrCb).astype(np.float32)
Y=ycc[...,0]
bl=cv2.GaussianBlur(Y,(0,0),1.6)
d=Y-bl
d=np.where(np.abs(d)<2,0,d)
Y=np.clip(Y+0.55*d,0,255)
# clamp overshoot to limit halos
loc_max=cv2.dilate(ycc[...,0],np.ones((5,5)));loc_min=cv2.erode(ycc[...,0],np.ones((5,5)))
ycc[...,0]=np.clip(Y,loc_min-6,loc_max+6)
final=cv2.cvtColor(np.clip(ycc,0,255).astype(np.uint8),cv2.COLOR_YCrCb2BGR)
rng=np.random.default_rng(1)
gr=cv2.GaussianBlur(rng.normal(0,1,final.shape[:2]).astype(np.float32),(0,0),0.7)*2.2
final=np.clip(final.astype(np.float32)+gr[...,None],0,255).astype(np.uint8)
fin=Image.fromarray(cv2.cvtColor(final,cv2.COLOR_BGR2RGB))
fin.save(out+'foto_final_4x5.jpg',quality=95,subsampling=0,optimize=True)
# before/after (original crop, upscaled same way, vs final)
o=orig.crop((0,y0,900,y0+h)).resize(fin.size,Image.LANCZOS)
s=Image.new('RGB',(fin.width*2+30,fin.height),'white');s.paste(o,(0,0));s.paste(fin,(fin.width+30,0))
s.thumbnail((2400,2400));s.save(out+'antes_depois.jpg',quality=90)
g=cv2.cvtColor(final,cv2.COLOR_BGR2GRAY)
print(fin.size,'mean',g.mean(),np.percentile(g,[1,50,99]),'clip hi',(g>250).mean(),'lo',(g<5).mean())
