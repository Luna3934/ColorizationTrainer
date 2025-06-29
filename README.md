# ColorizationTrainer

Deep Learning-Based Image Colorization
Developed a grayscale-to-color image colorization pipeline using a VGG16-based encoder-decoder architecture in PyTorch. The model predicts quantized ab color bins per pixel and applies confidence-weighted post-processing, including softmax temperature scaling, Gaussian smoothing, and custom color splotch suppression techniques. Achieved improved visual quality and PSNR scores by blending raw outputs with neighborhood-aware repairs and evaluating results through automated metrics and side-by-side image comparisons.