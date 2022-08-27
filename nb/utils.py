import requests
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import os
from astropy.io import fits # for opening fits file
import pandas as pd
from keras.preprocessing.image import img_to_array
from numpy import asarray




# Image helper functions
def gray(im):
    """
    Receive multi channels of an image and return a gray
    """
    return color.rgb2gray(im)
def red(im):
    """
    Receive multi channels of an image and return the red channel
    """
    return im[:,:,0]
def green(im):
    """
    Receive multi channels of an image and return the green channel
    """
    return im[:,:,1]
def blue(im):
    """
    Receive multi channels of an image and return the blue channel
    """
    return im[:,:,2]

def imread(filename,array=False):
    img = Image.open(filename)
    if array:
        img_arr = np.asarray(img).copy()
        img.close()
        return img_arr
    return img

def grad_x(img):
    """
    Naive implementation of gradient in x-direction
    """
    img2 = img.astype(np.float)
    dimg = np.zeros(img.shape).astype(np.float)
    for i in range(img2.shape[0]):
        y = img2[i,:]
        x = np.arange(img2.shape[1]).astype(np.float)
        dy = np.zeros(y.shape,np.float).astype(np.float)
        dy[0:-1] = np.diff(y)/np.diff(x)
        dy[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
        dimg[i] = dy
    return dimg
def grad_y(img):
    """
    Naive implementation of gradient in y-direction
    """
    img2 = img.astype(np.float)
    dimg = np.zeros(img.shape).astype(np.float)
    for i in range(img2.shape[1]):
        y = img2[:,i]
        x = np.arange(img2.shape[0]).astype(np.float)
        dy = np.zeros(y.shape,np.float).astype(np.float)
        dy[0:-1] = np.diff(y)/np.diff(x)
        dy[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
        dimg[:,i] = dy
    return dimg

def grad_mag(grad_x,grad_y):
    """
    returns the amount of change in the gradient
    """
    return np.sqrt(grad_x**2+grad_y**2)
def draw_h_line(imarray,p):
    """
    Receive 3 channel image and draw horizontal line at @p 
    """
    from skimage.draw import line_aa
    if len(imarray.shape) > 2:
        h,w,_ = imarray.shape
        value = [0,0,0]
    else:
        h,w = imarray.shape
        value = [0]
    rr, cc, val = line_aa(p, 0, p, w-1)
    imarray[rr,cc] = value

def draw_v_line(imarray,p):
    """
    Receive 3 channel image and draw vertical line at @p 
    """
    from skimage.draw import line_aa
    if len(imarray.shape) > 2:
        h,w,_ = imarray.shape
        value = [0,0,0]
    else:
        h,w = imarray.shape
        value = [0]
    h,w,_ = imarray.shape
    rr, cc, val = line_aa(0, p, h-1, p)
    imarray[rr,cc] = value 
    
    
def plot_row(imarray,index,ax=None):
    """
    Receive 1 channel of an image and plot row at @index 
    """
    y = imarray[index,:]
    x = np.arange(len(y))
    if ax is None:
        plt.plot(x,y)
    else:
        ax.plot(x,y)
# defining functions
def plot_col(imarray,index,ax=None):
    """
    Receive 1 channel of an image and plot col at @index 
    """
    y = imarray[:,index]
    x = np.arange(len(y))
    if ax is None:
        plt.plot(x,y)
    else:
        ax.plot(x,y)

def plot_grid(imarrays,titles,figsize=(10,10)):
    """
    show four images
    """
    count = 1
    plt.figure(figsize=figsize)
    for im,t in zip(imarrays,titles):
        plt.subplot(220+count)
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator()) 
        count += 1
        if len(im.shape) < 3:
            plt.imshow(im,cmap="gray")
        else:
            plt.imshow(im)
        plt.title(t, size=20)
    plt.gcf().set_tight_layout(True)
    plt.show()

def plot_row_with_image(img,rownum,ax1,ax2):
    img_copy = img.copy()
    if len(img_copy.shape) > 2:
        h,w,_ = img_copy.shape
        ax1.imshow(img_copy)
    else:
        h,w = img_copy.shape
        ax1.imshow(img_copy,cmap="gray")
    draw_h_line(img_copy,rownum)
    plot_row(img, rownum,ax2)
    
def normalize_img(s):
    """
    Receive image in any data types and range and normalize it
    to [0,255]
    """
    start = 0
    end = 255
    width = end - start
    res = (s - s.min())/(s.max() - s.min()) * width + start
    return res


def download_file_12(plate,mjd,fiber):
    link = "https://dr12.sdss.org/sas/dr12/sdss/spectro/redux/26/spectra/{0:04d}/spec-{1:04d}-{2:05d}-{3:04d}.fits"
    if os.path.exists(f'data/archive/{plate}_{mjd}_{fiber}.fits'):
        return
    tlink = link.format(plate,plate,mjd,fiber)
    print(tlink)
    r = requests.get(tlink)
    with open(f'data/archive/{plate}_{mjd}_{fiber}.fits', 'wb') as f:
        f.write(r.content)
        
# create a function to load fits and return needed data
def get_spectrum_data(filename):
    star = fits.open(filename)
    table_spec = star[1].data 
    table_line = star[3].data
    spectrum = pd.DataFrame(table_spec.tolist(),columns=table_spec.columns.names) 
    wavelines = pd.DataFrame(table_line.tolist(),columns=table_line.columns.names)
    spectrum["lam"] = 10**spectrum["loglam"]
    xspec, yspec = np.array(spectrum["lam"]), spectrum["flux"]    
    pltwavelines = wavelines[wavelines["LINEWAVE"] > xspec.min()]
    pltwavelines = pltwavelines[pltwavelines["LINEWAVE"] < xspec.max()]
    lineName, lineValue = pltwavelines["LINENAME"].tolist(), pltwavelines["LINEWAVE"].tolist()
    return xspec,yspec,lineName, lineValue

# create a function for plotting
from scipy.signal import savgol_filter
from matplotlib.pyplot import cm
def plot_spectrum(name,xspec,yspec,lineName,lineValue,ax):
    yspec = savgol_filter(yspec, 15, 5)
    
    minXSpec = xspec.min()
    maxXSpec = xspec.max()
    minYSpec = yspec.min()
    maxYSpec = yspec.max()
    
    # setting size of plot and limits
    #plt.gcf().set_size_inches((15,10))
    ax.set_xlim((minXSpec - 3, maxXSpec + 20))
    ax.set_ylim((minYSpec - 10, maxYSpec + 3))
    ax.set_title(name)

    # first plotting the spectrum
    ax.plot(xspec,yspec)

    # second plotting lines
    color = cm.rainbow(np.linspace(0,1,len(lineName)))[::-1]
    ccount = 0
    for name, value in zip(lineName,lineValue):
        # in case the name was bytes
        name = name.decode("utf-8").strip() 

        # plotting the line
        ax.axvline(value,0,1,ls="solid",label=name,c=color[ccount])
        ccount+=1
    ax.legend()
    

    
# download function for dr16
import requests
def download_file_16(plate,mjd,fiber):
    link = "https://dr16.sdss.org/optical/spectrum/view/data/format=fits/spec=lite?plateid={}&mjd={}&fiberid={}"
    print(plate,mjd,fiber)
    if os.path.exists(f'data/{plate}_{mjd}_{fiber}.fits'):
        return
    tlink = link.format(plate,mjd,fiber)
    print(tlink)
    r = requests.get(tlink)
    with open(f'data/{plate}_{mjd}_{fiber}.fits', 'wb') as f:
        f.write(r.content)
        
# function to extract wavelength, flux, class, and subclass of star from fits file
def get_spectrum_and_class(filename):
    print(filename)
    star = fits.open(filename)
    table_spec = star[1].data 
    table_info = star[2].data 
    starspectrum = pd.DataFrame(table_spec.tolist(),columns=table_spec.columns.names) 
    starinfo = pd.DataFrame(table_info.tolist(),columns=table_info.columns.names) 
    starspectrum["lam"] = 10**starspectrum["loglam"]
    xspec, yspec = np.array(starspectrum["lam"]), np.array(starspectrum["flux"])
    c = starinfo.values[0][61]
    c2 = starinfo.values[0][62]
    return xspec,yspec,c.decode('utf-8'),c2.decode('utf-8')



# convert plt figure to image
def to_pil(figure,dpi=128):
    buf = io.BytesIO()
    figure.savefig(buf,dpi=dpi,format='png',
    transparent=True,bbox_inches='tight',pad_inches=0)
    buf.seek(0)
    return Image.open(buf).resize((224,224))

# transforming spectrum data to spectrum image
def transform(spec,chunksize=100,chunk_width=10):
    new_size = int(spec.shape[0]/chunksize)*chunksize
    spectrum = spec[:new_size]
    groups = spectrum.reshape((-1,chunksize))
    f_groups = np.abs(fft(groups))
    plt.gcf().clear()
    plt.imshow(np.log(f_groups).T,aspect=0.1)
    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    return to_pil(plt.gcf())


def clean_data(df):
    # 1. stripping string trailing spaces
    df["subclass"] = df["subclass"].str.strip()
    # 2. filtering out bad entries
    df = df[df["subclass"]!= "Carbon_lines"] 
    df = df[df["subclass"] != ""]
    # 3. creating spectral type column as the first letter of the subclass
    df["spectral_type"] = df["subclass"].apply(lambda x: x[0])
    return df

def transform_imgs_to_arrays(images):
    # transforming images to arrays
    imgs_array = [img_to_array(img) for img in images]
    # assigning ML (input) to (image_as_array)
    input_images = asarray(imgs_array)
    # taking only first 3 channels in case of png
    input_images = input_images[:,:,:,:3]
    return input_images

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
# encoding labels to one-hot vector
le = preprocessing.LabelEncoder()
integer_encoded = None
def encode_target(spectral_type,classes):
    # classes is defined above from the distribution
    le.fit(classes)
    # tranform labels to integers
    integer_encoded = le.transform(spectral_type)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    # transform integer to onehot vector
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded



def split_data(inp,targ):
    # splitting data to training and testing
    training_index,testing_index = [],[]
    # contains a set of all data index
    all_index_set = set(np.arange(inp.shape[0]))
    # building class uniform training data
    # looping over targets, selecting %70 of the target data for training
    for i in range(targ.shape[1]):
        t_index = np.where(targ[:,i] == 1)[0]
        # selecting 70% of this class data for training
        tr_index = set(np.random.choice(t_index,int(t_index.size*0.7),replace=False))
        ts_index = all_index_set - tr_index
        training_index += list(tr_index)
        testing_index += list(ts_index)
    return training_index, testing_index


# function takes a set of data index, use the model to predict, and return predicted class and target class
def predict(model,ml_input,ml_targets=None):
    # clculate softmax
    softmax = model.predict(ml_input)
    # taking class with maximum probability for each index
    p_label_index = np.argmax(softmax,axis=1)
    # transform predicted class index to class letter
    p_labels = le.inverse_transform(p_label_index)
    # transforming target class index to class letter
    if ml_targets is not None:
        p_targets = le.inverse_transform(ml_targets.argmax(axis=1))
        return p_labels,p_targets
    else:
        return p_labels