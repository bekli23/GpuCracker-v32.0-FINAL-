# Bitcoin Recovery Tools - HTML Demo

## Overview

This is a **demonstration web interface** for the Bitcoin Recovery Tools suite, showcasing the functionality of:

- **SeedRecover** - BIP39 Mnemonic Recovery
- **GpuCracker** - High-Performance GPU Seed Cracker
- **Bloom Filter Builder** - Optimized Address Filter Creation
- **AKM Tools** - Advanced Key Management with 300+ Profiles

## Features

### 🎨 Design
- Modern dark theme with Bitcoin-inspired orange accents
- Matrix rain background animation
- 3D rotating cube visual in hero section
- Smooth animations and transitions
- Responsive design for all screen sizes

### 📱 Pages
1. **index.html** - Main landing page with overview
2. **SeedRecover.html** - Mnemonic recovery interface
3. **GpuCracker.html** - GPU cracking configuration
4. **build_bloom.html** - Bloom filter creation tool
5. **akm_seed2priv.html** - AKM profile explorer

### 🎭 Interactive Elements
- Animated counters
- Progress bars with shimmer effects
- Terminal-style output displays
- Profile card explorers
- Tabbed interfaces
- File drag-and-drop zones

## Usage

### Viewing the Demo

Simply open `index.html` in any modern web browser:

```bash
# Linux/Mac
open main_html/index.html

# Or navigate to the file in your browser
firefox main_html/index.html
chrome main_html/index.html
```

### File Structure
```
main_html/
├── index.html              # Main page
├── SeedRecover.html        # Seed recovery tool
├── GpuCracker.html         # GPU cracker interface
├── build_bloom.html        # Bloom filter builder
├── akm_seed2priv.html      # AKM tools
├── css/
│   ├── main.css           # Main styles
│   └── tool.css           # Tool-specific styles
├── js/
│   ├── matrix.js          # Matrix rain animation
│   ├── main.js            # Common functionality
│   ├── seedrecover.js     # SeedRecover demo
│   ├── gpucracker.js      # GpuCracker demo
│   ├── build_bloom.js     # Bloom filter demo
│   └── akm_tools.js       # AKM tools demo
└── README.md              # This file
```

## Technologies Used

- **HTML5** - Semantic markup
- **CSS3** - Grid, Flexbox, Animations
- **JavaScript** - Interactive functionality
- **Font Awesome** - Icons
- **Google Fonts** - JetBrains Mono & Inter

## Demo Limitations

⚠️ **Important**: This is a frontend demonstration only!

- No actual cryptographic operations
- No real Bitcoin key generation
- No actual GPU computation
- Simulated progress and results

For the real tools, compile and run the C++/CUDA applications.

## Browser Support

- ✅ Chrome/Edge 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Opera 76+

## Screenshots

The demo includes:
1. Animated matrix background
2. Glitch text effects
3. 3D rotating cube
4. Interactive terminal displays
5. Real-time progress simulation

## Credits

- Design inspired by modern cryptocurrency interfaces
- Matrix rain effect adapted from various open sources
- Icons by Font Awesome
- Fonts by Google Fonts

## License

This HTML demo is for educational purposes only.

---

**Note**: For the actual Bitcoin Recovery Tools, see the main project documentation.
