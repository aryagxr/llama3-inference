#!/usr/bin/env python3
"""
This script is vibecoded! Thanks to cursor.

Script to capture token generation output and create a GIF showing the exact speed.
Usage: python capture_output.py <file_to_run> [--max-tokens N] [--fps FPS]
"""



import subprocess
import sys
import time
import threading
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import argparse
import os

# Try to import imageio for GIF creation, fallback to PIL
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    # Don't print warning here - only if we actually need to create GIF

class OutputCapture:
    def __init__(self, max_tokens=50, fps=10):
        self.max_tokens = max_tokens
        self.fps = fps
        self.frame_duration = 1.0 / fps
        self.output_lines = []
        self.timestamps = []
        self.start_time = None
        self.lock = threading.Lock()
        self.running = True
        
    def start(self):
        self.start_time = time.time()
        
    def add_line(self, line):
        if self.start_time is None:
            self.start_time = time.time()
        timestamp = time.time() - self.start_time
        with self.lock:
            self.output_lines.append(line)
            self.timestamps.append(timestamp)
    
    def get_lines_at_time(self, t):
        """Get all lines that should be visible at time t"""
        with self.lock:
            visible_lines = []
            for i, timestamp in enumerate(self.timestamps):
                if timestamp <= t:
                    visible_lines.append(self.output_lines[i])
            return visible_lines
    
    def get_all_data(self):
        with self.lock:
            return list(zip(self.timestamps, self.output_lines))

def read_output(process, capture):
    """Read output from process in real-time"""
    try:
        for line in iter(process.stdout.readline, b''):
            if not capture.running:
                break
            line_str = line.decode('utf-8', errors='replace').rstrip()
            if line_str:
                capture.add_line(line_str)
                print(line_str, flush=True)  # Also print to console
    except Exception as e:
        print(f"Error reading output: {e}", file=sys.stderr)
    finally:
        capture.running = False

def find_font(size=14):
    """Try to find a monospace font"""
    font_paths = [
        '/System/Library/Fonts/Menlo.ttc',  # macOS
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',  # Linux
        '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',  # Linux
        'C:/Windows/Fonts/consola.ttf',  # Windows
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                pass
    
    # Fallback to default font
    try:
        return ImageFont.load_default()
    except:
        return None

def create_frame(lines, width=1200, height=800, font_size=14, padding=20, title_text=None):
    """Create a single frame showing the current output"""
    img = Image.new('RGB', (width, height), color='#0a0a0a')
    draw = ImageDraw.Draw(img)
    
    font = find_font(font_size)
    if font is None:
        font = ImageFont.load_default()
    
    title_font = find_font(font_size + 4)
    if title_font is None:
        title_font = font
    
    y = padding
    line_height = font_size + 6
    
    # Draw title
    title = title_text or "Token Generation Output"
    draw.text((padding, y), title, fill='#00ff00', font=title_font)
    y += int(line_height * 1.5)
    
    # Draw separator
    draw.line([(padding, y), (width - padding, y)], fill='#333333', width=2)
    y += line_height
    
    # Calculate how many lines can fit
    available_height = height - y - padding
    max_lines = int(available_height / line_height)
    
    # Show the most recent lines that fit (like a scrolling terminal)
    if len(lines) > max_lines:
        # Show indicator that there's more content above
        draw.text((padding, y), f"... ({len(lines) - max_lines} lines above) ...", fill='#666666', font=font)
        y += line_height
        # Show only the last max_lines-1 lines (minus 1 for the indicator)
        lines_to_show = lines[-(max_lines-1):]
    else:
        lines_to_show = lines
    
    # Draw visible lines
    for line in lines_to_show:
        if y + line_height > height - padding:
            break
        
        # Color code different types of lines
        if line.startswith("Step"):
            color = '#00ff88'  # Bright green for token steps
        elif "PERFORMANCE METRICS" in line or line.startswith("Time to") or line.startswith("Tokens"):
            color = '#ffff00'  # Yellow for metrics
        elif "FINAL GENERATED TEXT" in line or "=" in line:
            color = '#00aaff'  # Blue for headers
        else:
            color = '#cccccc'  # Light gray for other text
        
        # Truncate long lines
        max_chars = int((width - 2 * padding) / (font_size * 0.6))
        if len(line) > max_chars:
            line = line[:max_chars-3] + "..."
        
        draw.text((padding, y), line, fill=color, font=font)
        y += line_height
    
    return img

def create_gif_with_imageio(frames, output_path, fps=10):
    """Create GIF using imageio (better quality)"""
    imageio.mimsave(output_path, frames, fps=fps, loop=0)

def create_gif_with_pil(frames, output_path, fps=10):
    """Create GIF using PIL (fallback)"""
    duration_ms = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False
    )

def main():
    parser = argparse.ArgumentParser(description='Capture token generation output and create GIF')
    parser.add_argument('file', help='Python file to run (e.g., 01-naive.py)')
    parser.add_argument('--max-tokens', type=int, default=50, 
                       help='Maximum number of tokens to generate (default: 50)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for GIF (default: 10)')
    parser.add_argument('--width', type=int, default=1200,
                       help='GIF width in pixels (default: 1200)')
    parser.add_argument('--height', type=int, default=800,
                       help='GIF height in pixels (default: 800)')
    parser.add_argument('--font-size', type=int, default=14,
                       help='Font size (default: 14)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output GIF filename (default: <file>_output.gif)')
    
    args = parser.parse_args()
    
    # Resolve file path - try relative to current dir, then relative to script dir
    if not os.path.exists(args.file):
        # Try relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(script_dir, '..', args.file)
        if os.path.exists(potential_path):
            args.file = os.path.abspath(potential_path)
        elif not os.path.isabs(args.file):
            # Try from current working directory
            cwd_path = os.path.join(os.getcwd(), args.file)
            if os.path.exists(cwd_path):
                args.file = os.path.abspath(cwd_path)
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found")
        print(f"Current working directory: {os.getcwd()}")
        sys.exit(1)
    
    # Determine output filename
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        args.output = f"{base_name}_output.gif"
    
    print(f"Running {args.file} and capturing output...")
    print(f"Output will be saved to {args.output}")
    print("-" * 60)
    
    # Create capture object
    capture = OutputCapture(max_tokens=args.max_tokens, fps=args.fps)
    capture.start()
    
    # Set up environment for unbuffered output and optional max tokens
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering
    if args.max_tokens:
        env['MAX_NEW_TOKENS'] = str(args.max_tokens)
    
    # Start the process
    process = subprocess.Popen(
        [sys.executable, args.file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=os.path.dirname(os.path.abspath(args.file)) or '.'
    )
    
    # Start reading output in a separate thread
    reader_thread = threading.Thread(target=read_output, args=(process, capture))
    reader_thread.daemon = True
    reader_thread.start()
    
    # Wait for process to complete
    process.wait()
    capture.running = False
    reader_thread.join(timeout=5)
    
    # Get all captured data
    data = capture.get_all_data()
    
    if not data:
        print("No output captured!")
        sys.exit(1)
    
    print("\n" + "-" * 60)
    print(f"Captured {len(data)} lines of output")
    print("Creating GIF frames...")
    
    # Determine time range
    if data:
        max_time = data[-1][0]
        total_frames = int(max_time * args.fps) + 1
    else:
        total_frames = 1
        max_time = 1.0
    
    # Create frames - show text appearing at the exact speed it was generated
    frames = []
    if data:
        # Create frames based on actual timestamps
        max_time = data[-1][0]
        # Add some padding at the end to show final result
        total_duration = max_time + 2.0
        total_frames = int(total_duration * args.fps) + 1
        
        # Get base filename for title
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        title = f"{base_name} - Token Generation"
        
        for frame_idx in range(total_frames):
            t = frame_idx / args.fps
            visible_lines = capture.get_lines_at_time(t)
            frame = create_frame(visible_lines, args.width, args.height, args.font_size, title_text=title)
            frames.append(frame)
            
            if frame_idx % 10 == 0:
                print(f"Created {frame_idx + 1}/{total_frames} frames...", end='\r')
    else:
        # Fallback: create a single frame
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        title = f"{base_name} - Token Generation"
        frames.append(create_frame(["No output captured"], args.width, args.height, args.font_size, title_text=title))
    
    print(f"\nCreated {len(frames)} frames")
    print("Saving GIF...")
    
    # Save GIF
    try:
        if HAS_IMAGEIO:
            create_gif_with_imageio(frames, args.output, args.fps)
        else:
            print("Note: imageio not available. Using PIL (slower). Install with 'pip install imageio' for better performance.")
            create_gif_with_pil(frames, args.output, args.fps)
    except Exception as e:
        print(f"Error creating GIF: {e}")
        print("Trying alternative method...")
        try:
            create_gif_with_pil(frames, args.output, args.fps)
        except Exception as e2:
            print(f"Failed to create GIF: {e2}")
            sys.exit(1)
    
    print(f"✓ GIF saved to {args.output}")
    if data:
        print(f"  Duration: {data[-1][0]:.2f} seconds")
    print(f"  Frames: {len(frames)}")
    print(f"  FPS: {args.fps}")

if __name__ == '__main__':
    main()

