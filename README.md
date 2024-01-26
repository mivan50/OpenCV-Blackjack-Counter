# Blackjack Card Counter

## About the Project

This Blackjack Card Counter, built with Python and utilizing the OpenCV library, allows you to analyze card dealing videos or live video captures. The code can identify any card in view, keep track of the player and dealer hand totals, and provide a running count for card counting.

### Key Features:

- **Card Identification:** Uses OpenCV for precise card detection and recognition in both videos and live captures.
  
- **Hand Totals:** Keeps track of the player and dealer hand totals for Blackjack.
  
- **Running Count:** Implements card counting with a real-time running count feature.
  
- **Flexible Input:** Supports both video files and live video captures for versatile use.

## Preview:
![Preview_GIF](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYTNjMmU5cGVvaDhuaWhja2Vkc3dhbDF3Nm02OXBkOGp0bWtpNTloZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/ePnQprnoyDdsXUbNxX/giphy-downsized-large.gif)

## 🚀 Getting Started
1. **Requirements:**
   - Ensure Python is installed.

2. **Install Pygame:**
   - Install the required OpenCV library:

     ```bash
     pip install opencv-python
     ```

3. **Run the Script:**
   - Run the script with:

     ```bash
     python main.py
     ```
## 🎥 Customizing Video Input

By default, the script processes a video file. However, you can customize the input to either use a different video file or enable live video capture.

1. **Change Video File:**
   - Open `main.py` in a text editor.
   - Locate the `video_path` variable and update it with the path to your desired video file.

     ```python
     video_path = 'path/to/your/video/file.mp4'
     ```

2. **Use Live Video Capture:**
   - Open `main.py` in a text editor.
   - Set `video_path` to `0` to enable live video capture.

     ```python
     video_path = 0  # for live video capture
     ```

## 📥 Download
Clone the repository:

```bash
git clone https://github.com/mivan50/OpenCV-Blackjack-Counter.git
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
