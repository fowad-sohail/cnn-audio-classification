def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 20:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    return sample_width, r

def get_bounds(ds):
    np.array(ds)
    lds = len(ds)
    count = 0
    ll=-1
    ul=-1

    #Lower Limit
    for i in range(0,lds,WINDOW_SIZE):
        sum = 0
        for k in range(i,(i+WINDOW_SIZE)%lds):
            sum = sum + np.absolute(ds[k])
        if(sum>THRESHOLD):
            count +=1
        if(count>CHECK_THRESH):
            ll = i - WINDOW_SIZE * CHECK_THRESH
            break
        
    #Upper Limit
    count = 0
    for j in range(i,lds,WINDOW_SIZE):
        sum = 0
        for k in range(j,(j+WINDOW_SIZE)%lds):
            sum = sum + np.absolute(ds[k])
        if(sum<THRESHOLD):
            count +=1
        if(count>CHECK_THRESH):
            ul = j - WINDOW_SIZE * CHECK_THRESH


        if(ul>0 and ll >0):
            break
    return ll, ul 


def record_to_file(path):
    
    sample_width, data = record()
    ll, ul = get_bounds(data)
    print(ll,ul)
    if(ul-ll<100):
        return 0
    #nonz  = np.nonzero(data)
    ds = data[ll:ul]
    if(IS_PLOT):
        plt.plot(data)
        plt.axvline(x=ll)
        #plt.axvline(x=ll+5000)
        plt.axvline(x=ul)
        plt.show()

    #data = pack('<' + ('h'*len(data)), *data)
    fname = "0.wav"
    if not os.path.exists(path):
        os.makedirs(path)
    wf = wave.open(os.path.join(path,fname), 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(ds)
    wf.close()
    return 1

def findDuration(fname):
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        sw   = f.getsampwidth()
        chan = f.getnchannels()
        duration = frames / float(rate)
        #print("File:", fname, "--->",frames, rate, sw, chan)
        return duration

def graph_spectrogram(wav_file, nfft=512, noverlap=511):
    findDuration(wav_file)
    rate, data = wavfile.read(wav_file)
    #print("")
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=noverlap, NFFT=nfft)
    ax.axis('off')
    plt.rcParams['figure.figsize'] = [0.75,0.5]
    #fig.savefig('sp_xyz.png', dpi=300, frameon='false')
    fig.canvas.draw()
    size_inches  = fig.get_size_inches()
    dpi          = fig.get_dpi()
    width, height = fig.get_size_inches() * fig.get_dpi()

    #print(size_inches, dpi, width, height)
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #print("MPLImage Shape: ", np.shape(mplimage))
    imarray = np.reshape(mplimage, (int(height), int(width), 3))
    plt.close(fig)
    return imarray

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def normalize_gray(array):
    return (array - array.min())/(array.max() - array.min())

