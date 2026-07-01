#2024147_AYUSHEE_PCS

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

#BASIC BUILDING BLOCKS 
def generate_source(freq=50, fs_ct=50000, duration=0.06):
    t = np.arange(0, duration, 1/fs_ct)
    x = np.sin(2*np.pi*freq*t)
    return t, x

def sample_quantize(x, fs_ct, fs_sample, n_bits):
    step = int(fs_ct/fs_sample)
    x_s = x[::step]
    vmax = np.max(np.abs(x_s))*1.001
    L = 2**n_bits
    delta = 2*vmax/L
    idx = np.clip(np.floor((x_s + vmax)/delta), 0, L-1).astype(int)
    x_q = (idx + 0.5)*delta - vmax
    bits = ((idx[:,None] >> np.arange(n_bits-1,-1,-1)) & 1).astype(int).flatten()
    return x_s, x_q, bits

def line_encode(bits, code='polar_nrz', sps=10, A=1.0):
    n = len(bits); out=np.zeros(n*sps)
    if code=='polar_nrz': out[:] = np.repeat(np.where(bits, A, -A), sps)
    elif code=='unipolar_nrz': out[:] = np.repeat(np.where(bits, A, 0), sps)
    elif code=='bipolar_ami':
        sign=1
        for i,b in enumerate(bits):
            if b: out[i*sps:(i+1)*sps]=sign*A; sign*=-1
    elif code=='polar_rz':
        half=sps//2
        for i,b in enumerate(bits):
            v=A if b else -A; out[i*sps:i*sps+half]=v
    elif code=='manchester':
        half=sps//2
        for i,b in enumerate(bits):
            if b:
                out[i*sps:i*sps+half]=A; out[i*sps+half:(i+1)*sps]=-A
            else:
                out[i*sps:i*sps+half]=-A; out[i*sps+half:(i+1)*sps]=A
    return out

def raised_cosine(T=1.0,beta=0.5,sps=10,span=6):
    Ts=T/sps
    t=np.arange(-span*T, span*T+Ts/2, Ts)
    p=np.zeros_like(t)
    for k,ti in enumerate(t):
        if beta!=0 and np.isclose(abs(ti),T/(2*beta),atol=1e-12):
            p[k]=(np.pi/4)*np.sinc(1/(2*beta))
        else:
            num=np.sinc(ti/T)*np.cos(np.pi*beta*ti/T)
            den=1-(2*beta*ti/T)**2
            p[k]=num/den
    return t,p

def sinc_pulse(T=1.0,sps=10,span=6):
    Ts=T/sps
    t=np.arange(-span*T,span*T+Ts/2,Ts)
    return t,np.sinc(t/T)

def pulse_shape(symbols,pulse,sps):
    imp=np.zeros(len(symbols)*sps)
    imp[::sps]=symbols
    return np.convolve(imp,pulse,mode='full')

def awgn(sig,snr_db):
    Ps=np.mean(sig**2)
    snr_lin=10**(snr_db/10)
    Pn=Ps/snr_lin
    noise=np.sqrt(Pn)*np.random.randn(len(sig))
    return sig+noise,Pn

def matched_filter(rx,pulse):
    return np.convolve(rx,pulse[::-1],mode='full')

def sample_symbols(filtered,sps,n,offset):
    idx=offset+np.arange(n)*sps
    idx=idx[idx<len(filtered)]
    return filtered[idx]

def detect(samples,code='polar_nrz',A=1.0):
    if 'uni' in code: th=0.5*A
    elif 'bipolar' in code: return (np.abs(samples)>0.5*A).astype(int)
    else: th=0
    return (samples>th).astype(int)

def ber_calc(b1,b2):
    n=min(len(b1),len(b2))
    return np.sum(b1[:n]!=b2[:n])/n

def plot_eye(sig,sps,ax=None,title='',skip=0,max_tr=200):
    if ax is None: _,ax=plt.subplots()
    win=2*sps; usable=len(sig)-skip-win
    n_tr=min(max_tr,max(1,usable//sps))
    t=np.arange(win)/sps
    for i in range(n_tr):
        s=skip+i*sps
        ax.plot(t,sig[s:s+win],color='tab:blue',alpha=0.35,lw=0.6)
    ax.set_title(title)
    ax.set_xlabel('Time (symbol periods)')
    ax.set_ylabel('Amplitude')
    ax.grid(True,alpha=0.3)
    return ax

#MAIN DEMO
def main():
    np.random.seed(0)
    fs_ct,fs_sample,n_bits=50000,1000,4
    sps,beta,span=10,0.5,6
    A=1.0

    #PCM Sampling & Quantization
    t_ct,x_ct=generate_source()
    x_s,x_q,bits=sample_quantize(x_ct,fs_ct,fs_sample,n_bits)
    plt.figure(figsize=(8,5))
    plt.plot(t_ct*1000,x_ct,'k',label='Analog m(t)')
    plt.plot(np.arange(len(x_s))*1000/fs_sample,x_s,'go',label='sampled',markersize=5)
    plt.stem(np.arange(len(x_s))*1000/fs_sample,x_q,linefmt='r-',markerfmt='ro',basefmt=' ',label='quantized')
    plt.title('PCM Sampling and 4-bit Quantization')
    plt.xlabel('Time (ms)'); plt.ylabel('Amplitude')
    plt.legend(loc='upper right'); plt.grid(True)

    # Show first 100 bits of PCM stream
    plt.figure(figsize=(9,3))
    plt.step(np.arange(100), bits[:100], where='post', color='b')
    plt.ylim(-0.2,1.2)
    plt.xlabel('Bit index'); plt.ylabel('Bit')
    plt.title('First 100 bits of PCM stream (MSB first per sample)')
    plt.grid(True,alpha=0.3)

    #Line Code Waveforms
    demo=np.random.randint(0,2,18)
    codes=['polar_nrz','unipolar_nrz','bipolar_ami','polar_rz','manchester']
    fig,axs=plt.subplots(len(codes),1,figsize=(9,6),sharex=True)
    for ax,c in zip(axs,codes):
        s=line_encode(demo,c,sps,A)
        ax.plot(np.arange(len(s))/sps,s)
        ax.grid(True,alpha=0.3); ax.set_ylabel(c)
    axs[-1].set_xlabel('Time (bit periods)')
    fig.suptitle('Line-code Waveforms for same 18-bit pattern',y=0.995)

    #Four Pulse Shape Eyes
    Nbits=5000
    bits_demo=np.random.randint(0,2,Nbits)
    sym=np.where(bits_demo==1,A,-A)
    sig_nrz=line_encode(bits_demo,'polar_nrz',sps,A)
    sig_rz=line_encode(bits_demo,'polar_rz',sps,A)
    _,prc=raised_cosine(1.0,beta,sps,span)
    _,psinc=sinc_pulse(1.0,sps,span)
    sig_rc=pulse_shape(sym,prc,sps); skip_rc=(len(prc)-1)//2
    sig_sinc=pulse_shape(sym,psinc,sps); skip_sinc=(len(psinc)-1)//2

    fig,axs=plt.subplots(2,2,figsize=(10,7))
    plot_eye(sig_nrz,sps,axs[0,0],'NRZ Rect')
    plot_eye(sig_rz,sps,axs[0,1],'RZ Rect')
    plot_eye(sig_rc,sps,axs[1,0],'Raised-Cosine β=0.5',skip_rc)
    plot_eye(sig_sinc,sps,axs[1,1],'Sinc Pulse',skip_sinc)
    fig.tight_layout(pad=2.0); fig.subplots_adjust(top=0.90)
    fig.suptitle('Binary Eye Diagrams for Four Pulse Shapes',y=0.99)

    #Raised‑cosine pulse plots
    T=1.0
    t_rc,p0=raised_cosine(T,beta,sps,span)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for b in (0.0,0.25,0.5,1.0):
        _,p=raised_cosine(T,b,sps,span)
        ax[0].plot(t_rc/T,p,label=('sinc(β=0)' if b==0 else f'RC β={b}'))
        P=np.fft.fftshift(np.fft.fft(p,4096))
        f=np.fft.fftshift(np.fft.fftfreq(len(P),d=T/sps))
        PdB=20*np.log10(np.abs(P)/np.max(np.abs(P))+1e-12)
        ax[1].plot(f,PdB,label=('sinc(β=0)' if b==0 else f'RC β={b}'))
    ax[0].set_title('Raised cosine Pulse (Time Domain)'); ax[0].grid(True); ax[0].legend()
    ax[1].set_xlim(-1.5,1.5); ax[1].set_ylim(-80,5)
    ax[1].set_title('Raised cosine vs Sinc (Frequency Domain)')
    ax[1].grid(True); ax[1].legend()

    #Eyes for different β
    betas=[0.0,0.25,0.5,1.0]
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    for ax,b in zip(axs.flatten(),betas):
        _,p=raised_cosine(1.0,b,sps,span)
        sig=pulse_shape(sym,p,sps); skip=(len(p)-1)//2
        plot_eye(sig,sps,ax=ax,title=f'Eye (β={b})',skip=skip)
    fig.tight_layout(pad=2.0); fig.subplots_adjust(top=0.90)
    fig.suptitle('Eye Diagrams for Various Raised-Cosine Roll-off Factors',y=0.99)

    #Polar NRZ noise‑free eye
    bits_nf=np.random.randint(0,2,4000)
    sig_nf=line_encode(bits_nf,'polar_nrz',sps,A)
    plt.figure(figsize=(7,4))
    plot_eye(sig_nf,sps,title='Eye diagram - Polar NRZ Rectangular Pulse (noise-free)',skip=sps//2)

    #Eye after AWGN + Matched Filter
    snr_db=8
    tx=sig_rc; rx,Pn=awgn(tx,snr_db)
    Ps=np.mean(tx**2)
    skip_tx=(len(prc)-1)//2
    plot_eye(rx,sps,title=f'Eye after AWGN ({snr_db}dB)',skip=skip_tx)

    mfo=matched_filter(rx,prc); skip_mf=len(prc)-1
    plot_eye(mfo,sps,title='Eye after Matched Filter',skip=skip_mf)

    smp=sample_symbols(mfo,sps,len(sym),skip_mf)
    det=detect(smp,'polar_nrz',A)
    ber=ber_calc(bits_demo,det)
    errs=np.sum(bits_demo[:len(det)]!=det); n=len(det)
    print(f"\nAWGN SNR={snr_db}dB  signalpower={Ps:.4f}  noise power={Pn:.4f}")
    print(f"Detected bits={n}, errors={errs}, BER={ber:.3e}")

    #Transmitted vs Detected bits
    plt.figure(figsize=(9,5))
    Nshow=60; ix=np.arange(Nshow)
    plt.subplot(2,1,1)
    plt.stem(ix,bits_demo[:Nshow],basefmt=' ',linefmt='b-',markerfmt='bo',label='transmitted')
    plt.ylim(-0.2,1.2); plt.title(f'Transmitted vs. Detected Bits (BER={ber:.3e})')
    plt.legend(); plt.grid(True,alpha=0.3)

    plt.subplot(2,1,2)
    plt.stem(ix,det[:Nshow],basefmt=' ',linefmt='r-',markerfmt='ro',label='detected')
    err=np.where(bits_demo[:Nshow]!=det[:Nshow])[0]
    plt.plot(err,det[err],'kx',ms=14,mew=2.5,label='error')
    plt.ylim(-0.2,1.2); plt.xlabel('bit index'); plt.grid(True,alpha=0.3)
    plt.legend(); plt.tight_layout()

    #BER vs SNR (Waterfall)
    Nbits_big = 1000000
    print(f"\nSimulating {Nbits_big:,} bits for Waterfall Curve")

    bits_big = np.random.randint(0,2,Nbits_big)
    symb = np.where(bits_big==1, A, -A)

    #Transmitter
    tx_big = pulse_shape(symb, prc, sps)

    snrs = np.arange(-4,13,2)
    skip_mf = len(prc)-1
    bers = []

    for snr in snrs:
        rx_big,_ = awgn(tx_big, snr)
        mf_big = matched_filter(rx_big, prc)
        smp_big = sample_symbols(mf_big, sps, len(symb), skip_mf)
        det_big = detect(smp_big, 'polar_nrz', A)

        ber_big = ber_calc(bits_big, det_big)
        bers.append(ber_big)

        print(f"SNR={snr:>2} dB --> BER={ber_big:9.6e}")


    #Convert & fix zeros
    bers = np.array(bers)
    bers[bers == 0] = 1e-7   # prevent log(0)

    #Theoretical BER
    snr_lin = 10**(snrs/10)
    ber_theory = 0.5 * erfc(np.sqrt(sps * snr_lin/2))

    if len(smp_big) == 0:
        print("ERROR: No samples extracted!")   
    

    #Plot Waterfall
    plt.figure(figsize=(7,5))

    plt.semilogy(snrs, bers, 'o-b', lw=2, ms=6, label='Simulated BER')
    plt.semilogy(snrs, ber_theory, 'r--', lw=2, label='Theoretical BER')

    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (log scale)')
    plt.title('BER vs SNR - Waterfall Curve')
    plt.grid(True, which='both')
    plt.legend()
    plt.ylim([1e-6, 1])

    plt.show()


    plt.show()

if __name__=="__main__":
    main()
