

```python
%matplotlib inline
%reload_ext autoreload
%autoreload 2

from fastai.imports import *
from fastai.core import *
from fastai.io import *
from fastai.dataloader import *
from fastai.conv_learner import *
from fastai.learner import *
from fastai.models.resnet import *
import os
from audio_dataset import *
from audio_transforms import *

import IPython.display as ipd
```


```python
import librosa
from librosa import display
```

## Load Data


```python
PATH = Path('data/freesound')
TRN_PATH = PATH/'audio_train'
TEST_PATH = PATH/'audio_test'
```


```python
trn = pd.read_csv(PATH/'train.csv')
test = pd.read_csv(PATH/'sample_submission.csv')
```


```python
verified = list(trn['manually_verified'])
```


```python
trn = trn[['fname','label']].copy()
trn_sample = trn[:1900]
trn.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fname</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00044347.wav</td>
      <td>Hi-hat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>001ca53d.wav</td>
      <td>Saxophone</td>
    </tr>
    <tr>
      <th>2</th>
      <td>002d256b.wav</td>
      <td>Trumpet</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0033e230.wav</td>
      <td>Glockenspiel</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00353774.wav</td>
      <td>Cello</td>
    </tr>
  </tbody>
</table>
</div>




```python
trn.to_csv('trn.csv', index=False)
trn_sample.to_csv('trn_sample.csv', index=False)
```


```python
trn.shape[0], len(trn.label.unique())
```




    (9473, 41)




```python
fnames = list(trn['fname']) 
test_fnames = list(test['fname']) 
len(fnames), len(test_fnames)
```




    (9473, 9400)




```python
trn_wavs = (PATH/'audio_train').glob('*.wav')
test_wavs = (PATH/'audio_test').glob('*.wav')
```


```python
stats = (np.array([ 0.18637]), np.array([ 0.30634]))
```

### Listen to Sounds


```python
test_preds = list(test.label[:10])
```


```python
length= int(3*44100) #seconds * sample_rate
n = 4

#play sample with stats
#length = 3*44100
#sample = os.path.join(TRN_PATH, fnames[n])
sample = os.path.join(TEST_PATH, test_fnames[n])
print(test_fnames[n])
raw = open_audio(sample)
raw_len = len(raw)
raw_s = adj_length(raw, length)
#print('raw length: ', raw_len, 'sample length:', len(raw_s))
#print('label:', trn['label'].iloc[n], 'verified:', verified[n])
print('prediction:', test_preds[n])
ipd.Audio(raw_s, rate=44100)
```

### Audio_transforms

Transforms taken from Fastai's `transforms.py` and modified to work with Audio files. *Not shown because competition is still in progress.


```python
# @hidden_cell
class Transform():
    """ A class that represents a transform.

    All other transforms should subclass it. All subclasses should override
    do_transform.

    Arguments
    ---------
        tfm_y : TfmType
            type of transform
    """
    def __init__(self, tfm_y=TfmType.NO):
        self.tfm_y=tfm_y
        self.store = threading.local()

    def set_state(self): pass
    def __call__(self, x, y):
        self.set_state()
        x,y = ((self.transform(x),y) if self.tfm_y==TfmType.NO
                else self.transform(x,y) if self.tfm_y in (TfmType.PIXEL, TfmType.CLASS)
                else self.transform_coord(x,y))
        return x, y

    def transform_coord(self, x, y): return self.transform(x),y

    def transform(self, x, y=None):
        x = self.do_transform(x,False)
        return (x, self.do_transform(y,True)) if y is not None else x

    @abstractmethod
    def do_transform(self, x, is_y): raise NotImplementedError


class Denormalize():
    """ De-normalizes an image, returning it to original format.
    """
    def __init__(self, m, s):
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
    def __call__(self, x): return x*self.s+self.m


class Normalize():
    """ Normalizes an image to zero mean and unit standard deviation, given the mean m and std s of the original image """
    def __init__(self, m, s): #tfm_y=TfmType.NO
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
        #self.tfm_y=tfm_y

    def __call__(self, x, y=None):
        x = (x-self.m)/self.s
        #if self.tfm_y==TfmType.PIXEL and y is not None: y = (y-self.m)/self.s
        return x,y

class ChannelOrder():
    '''
    changes image array shape from (h, w, 3) to (3, h, w). 
    tfm_y decides the transformation done to the y element. 
    '''
    def __init__(self, tfm_y=TfmType.NO): self.tfm_y=tfm_y

    def __call__(self, x, y):
        x = np.rollaxis(x, 2)
        #if isinstance(y,np.ndarray) and (len(y.shape)==3):
        #if self.tfm_y==TfmType.PIXEL: y = np.rollaxis(y, 2)
        #elif self.tfm_y==TfmType.CLASS: y = y[...,0]
        return x,y

def vocode(x,y,rate=2.0):
    return librosa.phase_vocoder(x, rate), y

def rand0(s): return random.random()*(s*2)-s

def rand1(s): return int(random.random()*s)

def focus_mel(aud, b, c):
    ''' highlights audio's mel_bands'''
    if b == 0: return aud
    mu = np.average(aud[:b])
    return aud[:b]+mu*c

class RandomFocus_mel(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b,self.c = b,c
        
    def set_state(self):
        self.store.b_rand = rand1(self.b)
        self.store.c_rand = self.c
        
    def do_transform(self, x, is_y):
        b = self.store.b_rand
        c = self.store.c_rand
        x = focus_mel(x, b, c)
        return x

def lighting(im, b, c):
    ''' adjusts image's balance and contrast'''
    if b==0 and c==1: return im
    mu = np.average(im)
    return np.clip((im-mu)*c+mu+b,0.,1.).astype(np.float32)

class RandomLighting(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b,self.c = b,c

    def set_state(self):
        self.store.b_rand = rand0(self.b)
        self.store.c_rand = rand0(self.c)

    def do_transform(self, x, is_y):
        #if is_y and self.tfm_y != TfmType.PIXEL: return x
        b = self.store.b_rand
        c = self.store.c_rand
        c = -1/(c-1) if c<0 else c+1
        x = lighting(x, b, c)
        return x
       
def compose(im, y, fns):
    """ apply a collection of transformation functions fns to images
    """
    for fn in fns:
        #pdb.set_trace()
        im, y =fn(im, y)
    return im if y is None else (im, y)


class Transforms():
    def __init__(self, sz, tfms, normalizer, denorm, crop_type=CropType.CENTER,
                 tfm_y=TfmType.NO, sz_y=None):
        if sz_y is None: sz_y = sz
        self.sz,self.denorm,self.norm,self.sz_y = sz,denorm,normalizer,sz_y
        crop_tfm = crop_fn_lu[crop_type](sz, tfm_y, sz_y)
        self.tfms = tfms
        self.tfms.append(crop_tfm)
        if normalizer is not None: self.tfms.append(normalizer)
        self.tfms.append(ChannelOrder(tfm_y))

    def __call__(self, im, y=None): return compose(im, y, self.tfms)
    def __repr__(self): return str(self.tfms)


def image_gen(normalizer, denorm, sz, tfms=None, max_zoom=None, pad=0, crop_type=None,
              tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT, scale=None):
    """
    Generate a standard set of transformations

    Arguments
    ---------
     normalizer :
         image normalizing function
     denorm :
         image denormalizing function
     sz :
         size, sz_y = sz if not specified.
     tfms :
         iterable collection of transformation functions
     max_zoom : float,
         maximum zoom
     pad : int,
         padding on top, left, right and bottom
     crop_type :
         crop type
     tfm_y :
         y axis specific transformations
     sz_y :
         y size, height
     pad_mode :
         cv2 padding style: repeat, reflect, etc.

    Returns
    -------
     type : ``Transforms``
         transformer for specified image operations.

    See Also
    --------
     Transforms: the transformer object returned by this function
    """
    if tfm_y is None: tfm_y=TfmType.NO
    if tfms is None: tfms=[]
    elif not isinstance(tfms, collections.Iterable): tfms=[tfms]
    if sz_y is None: sz_y = sz
    if scale is None:
        scale = [RandomScale(sz, max_zoom, tfm_y=tfm_y, sz_y=sz_y) if max_zoom is not None
                 else Scale(sz, tfm_y, sz_y=sz_y)]
    elif not is_listy(scale): scale = [scale]
    if pad: scale.append(AddPadding(pad, mode=pad_mode))
    if crop_type!=CropType.GOOGLENET: tfms=scale+tfms
    return Transforms(sz, tfms, normalizer, denorm, crop_type,
                      tfm_y=tfm_y, sz_y=sz_y)

def noop(x):
    """dummy function for do-nothing.
    equivalent to: lambda x: x"""
    return x

class AudTransforms():
    def __init__(self, tfms, normalizer, denorm):
        #if sz_y is None: sz_y = sz
        #self.sz,self.denorm,self.norm,self.sz_y = sz,denorm,normalizer,sz_y
        self.denorm,self.norm = denorm,normalizer
        #pdb.set_trace()
        #crop_tfm = crop_fn_lu[crop_type](sz, tfm_y, sz_y)
        self.tfms = tfms
        #self.tfms.append(crop_tfm)
        if normalizer is not None: self.tfms.append(normalizer)
        #self.tfms.append(ChannelOrder())

    def __call__(self, im, y=None): return compose(im, y, self.tfms) 
    def __repr__(self): return str(self.tfms)

def audio_gen(normalizer, denorm, tfms=None):
    if tfms is None: tfms = []
    elif not isinstance(tfms, collections.Iterable): tfms=[tfms]
    return AudTransforms(tfms, normalizer, denorm)

def aud_tfms_from_stats(stats, aug_tfms=None):
#def tfms_from_stats(stats, sz, aug_tfms=None, max_zoom=None, pad=0, crop_type=CropType.RANDOM,
                    #tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT, norm_y=True, scale=None):
    """ Given the statistics of the training image sets, returns separate training and validation transform functions
    """
    if aug_tfms is None: aug_tfms=[]
    #tfm_norm = Normalize(*stats, tfm_y=tfm_y if norm_y else TfmType.NO) if stats is not None else None
    tfm_norm = Normalize(*stats) if stats is not None else None
    tfm_denorm = Denormalize(*stats) if stats is not None else None
    #val_crop = CropType.CENTER if crop_type in (CropType.RANDOM,CropType.GOOGLENET) else crop_type
    #val_tfm = image_gen(tfm_norm, tfm_denorm, sz, pad=pad, crop_type=val_crop,
            #tfm_y=tfm_y, sz_y=sz_y, scale=scale)
    val_tfm = audio_gen(tfm_norm, tfm_denorm)
    #trn_tfm = image_gen(tfm_norm, tfm_denorm, sz, pad=pad, crop_type=crop_type,
            #tfm_y=tfm_y, sz_y=sz_y, tfms=aug_tfms, max_zoom=max_zoom, pad_mode=pad_mode, scale=scale)
    trn_tfm = audio_gen(tfm_norm, tfm_denorm, tfms=aug_tfms)
    return trn_tfm, val_tfm

```

### ResNet

Fastai's ResNet model.


```python
# @hidden_cell

def conv(ni, nf, ks=3, stride=1):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=False)


def bn1(planes):
    m = nn.BatchNorm1d(planes)
    m.weight.data.fill_(1)
    m.bias.data.zero_()
    return m

def bn(planes, init_zero=False):
    m = nn.BatchNorm2d(planes)
    m.weight.data.fill_(0 if init_zero else 1)
    m.bias.data.zero_()
    return m

class fc1(nn.Module):
    def __init__(self, ni, nf, ks=2, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(ni,nf,kernel_size=ks,stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.max = nn.MaxPool2d(2, stride=2, padding=1)
    
    def forward(self,x):
        out = self.conv(x)
        #return self.relu(out)
        out = self.relu(out)
        out = self.max(out)
        return out

class fc2(nn.Module):
    def __init__(self, ni, nf, ks=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(ni,nf,kernel_size=ks,stride=stride)
        self.sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):
        out = self.conv(x)
        return self.sigmoid(out)
        #return self.relu(out)

class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        #pdb.set_trace()
        return self.lambd(x)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv(inplanes, planes, stride=stride)
        self.bn1 = bn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes)
        self.bn2 = bn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None: residual = self.downsample(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)

        out = residual + out
        out = self.relu(out)
        out = self.bn2(out)
        
        return out
    

class MyResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, k=1, vgg_head=False):
        super().__init__()
        self.inplanes = 64

        features = [conv(1, 64, ks=7, stride=2)
            , bn(64) , nn.ReLU(inplace=True) , nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            , self._make_layer(block, int(64*k), layers[0])
            , self._make_layer(block, int(128*k), layers[1], stride=2)
            , self._make_layer(block, int(256*k), layers[2], stride=2)
            , self._make_layer(block, int(512*k), layers[3], stride=2)]
        out_sz = int(512*k) * block.expansion

        if vgg_head:
            features += [nn.AdaptiveAvgPool2d(3), Flatten()
                , nn.Linear(out_sz*3*3, 4096), nn.ReLU(inplace=True), bn1(4096), nn.Dropout(0.25)
                , nn.Linear(4096,   4096), nn.ReLU(inplace=True), bn1(4096), nn.Dropout(0.25)
                , nn.Linear(4096, num_classes)]
        else: features += [nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(out_sz, num_classes)]
        
        self.features = nn.Sequential(*features)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(self.inplanes, planes*block.expansion, ks=1, stride=stride),
                bn(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)#, nn.Dropout2d(0.5))

    def forward(self, x): return self.features(x)

```


```python
# from John Hartquist
def mapk_np(preds, targs, k=3):
    preds = np.argsort(-preds, axis=1)[:, :k]
    score = 0.0
    for i in range(k):
        num_hits = (preds[:, i] == targs).sum()
        score += num_hits * (1.0 / (i+1.0))
    score /= preds.shape[0]
    return score

def mapk(preds, targs, k=3):
    return mapk_np(to_np(preds), to_np(targs), k)
```


```python
def B6(ni,nf):
    return nn.Sequential(
        conv(ni,nf), 
        bn(nf), 
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2,2,padding=1))

class AudioCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        #self.num_classes = num_classes

        features = [BasicBlock(1,16), BasicBlock(16,32), BasicBlock(32,64),
                    BasicBlock(64,128), BasicBlock(128,256), B6(256,512),
                    fc1(512,1024), fc2(1024,num_classes), 
                    Lambda(lambda x: x.view(x.shape[0], 41, -1)), 
                    Lambda(lambda x: torch.mean(x, dim=2))]
        
        self.features = nn.Sequential(*features)
        
    def forward(self, x): return self.features(x)
        
```

### Model


```python
tfms = aud_tfms_from_stats(stats, aug_tfms=[RandomLighting(0.5,0.5)])
md = AudioClassifierData.from_csv(PATH, 'audio_train', 'trn.csv', val_idxs=1, bs=32, tfms=tfms, test_name='audio_test')
```


```python
m = MyResNet(BasicBlock, [3, 4, 6, 3], num_classes=41, vgg_head=False)
opt = optim.Adam
metrics = [accuracy, mapk]
loss = F.cross_entropy
learn = ConvLearner.from_model_data(m, md, crit=loss, metrics=metrics, opt_fn=opt)
```


```python
learn.unfreeze()
```


```python
learn.lr_find()
```


```python
learn.sched.plot()
```


```python
learn.fit(1e-4, 1, wds=1e-5, cycle_len=20, use_clr_beta=(5,20,0.95,0.75))
```


```python
learn.save('2d_res_1ch_hop25610s_3')
```


```python
learn.load('2d_res_1ch_hop25610s_3')
```

### Model Evaluation


```python
learn.model.eval()
val_preds = learn.predict_with_targs()

val_acc = accuracy_np(*val_preds)
val_map = mapk_np(*val_preds)

print(f'Val Acc: {val_acc:.3f}, Val MAP: {val_map:.3f}')
```

    Val Acc: 0.697, Val MAP: 0.770


### Predictions


```python
multi_preds, y = learn.TTA(is_test=True)
```

                                                  


```python
preds = np.mean(multi_preds, 0)
```


```python
np.save(PATH/'tmp/preds14.npy', preds)
```


```python
classes = np.array(sorted(trn.label.unique()))
top_3_idx = [np.argsort(preds[i])[-3:][::-1] for i in range(len(test_fnames))]
pred_labels = [list(classes[[top_3_idx[i]]]) for i in range(len(test_fnames))]
preds = [" ".join(ls) for ls in pred_labels]
preds[:5]
```




    ['Trumpet Saxophone Oboe',
     'Hi-hat Chime Shatter',
     'Cello Double_bass Acoustic_guitar',
     'Trumpet Violin_or_fiddle Meow',
     'Bass_drum Knock Gunshot_or_gunfire']




```python
tested = [md.test_ds.fnames[i].split('/')[-1] for i in range(len(test_fnames))]
```


```python
idx = []
for fname in test_fnames:
    for name in tested:
        if name == fname:
            idx.append(tested.index(name))
```


```python
[tested[i] for i in idx[:5]]
```




    ['00063640.wav',
     '0013a1db.wav',
     '002bb878.wav',
     '002d392d.wav',
     '00326aa9.wav']




```python
test_fnames[:5]
```




    ['00063640.wav',
     '0013a1db.wav',
     '002bb878.wav',
     '002d392d.wav',
     '00326aa9.wav']




```python
test['label'] = [preds[i] for i in idx]
```


```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fname</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00063640.wav</td>
      <td>Shatter Tearing Fireworks</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0013a1db.wav</td>
      <td>Flute Oboe Trumpet</td>
    </tr>
    <tr>
      <th>2</th>
      <td>002bb878.wav</td>
      <td>Bass_drum Computer_keyboard Knock</td>
    </tr>
    <tr>
      <th>3</th>
      <td>002d392d.wav</td>
      <td>Bass_drum Flute Cello</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00326aa9.wav</td>
      <td>Oboe Clarinet Telephone</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.to_csv(PATH/'tmp/sub13.csv', index=False)
```


```python
test.shape
```




    (9400, 2)


