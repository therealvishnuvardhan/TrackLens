# Generate synthetic test fixtures representing the title slide progression
Add-Type -AssemblyName System.Drawing

$imgDir = Join-Path $PSScriptRoot "images"
if (-not (Test-Path $imgDir)) {
    New-Item -ItemType Directory -Path $imgDir -Force | Out-Null
}

function Save-BmpAsPng($bmp, $path) {
    $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
}

# 1. v1_photographer_blog.png: 1200x800, Rich detail, stormy sky, floodwaters, rescue boat
Write-Host "Generating v1_photographer_blog.png..."
$bmp1 = New-Object System.Drawing.Bitmap(1200, 800)
$g1 = [System.Drawing.Graphics]::FromImage($bmp1)
$g1.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias

# Sky gradient (dark stormy grey-blue)
$skyRect = New-Object System.Drawing.Rectangle(0, 0, 1200, 420)
$skyBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
    $skyRect,
    [System.Drawing.Color]::FromArgb(40, 55, 75),
    [System.Drawing.Color]::FromArgb(90, 110, 130),
    [System.Drawing.Drawing2D.LinearGradientMode]::Vertical
)
$g1.FillRectangle($skyBrush, $skyRect)

# Clouds
$cloudBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(80, 200, 210, 220))
$g1.FillEllipse($cloudBrush, 150, 80, 400, 160)
$g1.FillEllipse($cloudBrush, 450, 60, 500, 180)
$g1.FillEllipse($cloudBrush, 850, 100, 320, 140)

# Distant houses/cityscape silhouettes
$buildingBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(50, 60, 70))
$g1.FillRectangle($buildingBrush, 80, 280, 120, 140)
$g1.FillRectangle($buildingBrush, 220, 240, 140, 180)
$g1.FillRectangle($buildingBrush, 380, 300, 90, 120)
$g1.FillRectangle($buildingBrush, 900, 260, 180, 160)

# Flood waters gradient (muddy reflective blue-brown)
$waterRect = New-Object System.Drawing.Rectangle(0, 400, 1200, 400)
$waterBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
    $waterRect,
    [System.Drawing.Color]::FromArgb(75, 95, 105),
    [System.Drawing.Color]::FromArgb(50, 70, 80),
    [System.Drawing.Drawing2D.LinearGradientMode]::Vertical
)
$g1.FillRectangle($waterBrush, $waterRect)

# Water ripples
$ripplePen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(60, 255, 255, 255), 2)
for ($y = 440; $y -lt 780; $y += 35) {
    $g1.DrawArc($ripplePen, 100, $y, 300, 20, 0, 180)
    $g1.DrawArc($ripplePen, 500, $y + 10, 400, 25, 0, 180)
    $g1.DrawArc($ripplePen, 800, $y - 10, 350, 20, 0, 180)
}

# Rescue Boat (Orange inflatable dinghy in center)
$boatBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(235, 90, 25))
$g1.FillEllipse($boatBrush, 450, 520, 340, 120)
$boatInnerBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(160, 50, 15))
$g1.FillEllipse($boatInnerBrush, 475, 535, 290, 80)

# Volunteer figures in boat with yellow life vests
# Volunteer 1 (Left - Rescuer with flashlight/oar)
$figure1Head = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(220, 180, 150))
$g1.FillEllipse($figure1Head, 510, 465, 45, 50)
$vestBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(245, 210, 40))
$g1.FillRectangle($vestBrush, 495, 510, 75, 60)

# Volunteer 2 (Center - Subject)
$figure2Head = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(200, 160, 130))
$g1.FillEllipse($figure2Head, 600, 455, 50, 55)
$g1.FillRectangle($vestBrush, 585, 505, 80, 65)

# Water splashes
$splashPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(180, 220, 240, 255), 3)
$g1.DrawLine($splashPen, 440, 590, 420, 570)
$g1.DrawLine($splashPen, 450, 600, 430, 620)
$g1.DrawLine($splashPen, 790, 590, 820, 570)

# Subtle signature watermark in corner
$fontWatermark = New-Object System.Drawing.Font("Arial", 12, [System.Drawing.FontStyle]::Italic)
$watermarkBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(140, 255, 255, 255))
$g1.DrawString("© 2024 Elena Rossi Photography - Valencia Floods", $fontWatermark, $watermarkBrush, 20, 765)

$g1.Dispose()
$v1Path = Join-Path $imgDir "v1_photographer_blog.png"
Save-BmpAsPng $bmp1 $v1Path
Write-Host "Created $v1Path"

# 2. v2_twitter.png: 1000x667 crop (100px from left/right, 66px from top/bottom) of v1
Write-Host "Generating v2_twitter.png..."
$bmp2 = New-Object System.Drawing.Bitmap(1000, 667)
$g2 = [System.Drawing.Graphics]::FromImage($bmp2)
$g2.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
# Crop rectangle inside v1: (100, 66, 1000, 667)
$srcRect2 = New-Object System.Drawing.Rectangle(100, 66, 1000, 667)
$destRect2 = New-Object System.Drawing.Rectangle(0, 0, 1000, 667)
$g2.DrawImage($bmp1, $destRect2, $srcRect2, [System.Drawing.GraphicsUnit]::Pixel)
$g2.Dispose()
$v2Path = Join-Path $imgDir "v2_twitter.png"
Save-BmpAsPng $bmp2 $v2Path
Write-Host "Created $v2Path"

# 3. v3_whatsapp.png: 800x533, resized from v2, red banner and meme text overlay added
Write-Host "Generating v3_whatsapp.png..."
$bmp3 = New-Object System.Drawing.Bitmap(800, 533)
$g3 = [System.Drawing.Graphics]::FromImage($bmp3)
$g3.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::Low
$destRect3 = New-Object System.Drawing.Rectangle(0, 0, 800, 533)
$srcRect3 = New-Object System.Drawing.Rectangle(0, 0, 1000, 667)
$g3.DrawImage($bmp2, $destRect3, $srcRect3, [System.Drawing.GraphicsUnit]::Pixel)

# Add Disinformation Red Banner across top
$bannerBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(210, 190, 20, 20))
$g3.FillRectangle($bannerBrush, 0, 0, 800, 55)
$fontBanner = New-Object System.Drawing.Font("Impact", 22, [System.Drawing.FontStyle]::Bold)
$whiteBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
$sf = New-Object System.Drawing.StringFormat
$sf.Alignment = [System.Drawing.StringAlignment]::Center
$g3.DrawString("BREAKING: SOUTHERN CALIFORNIA FLOODS", $fontBanner, $whiteBrush, 400, 10, $sf)

# Add WhatsApp Forward tag
$fontSub = New-Object System.Drawing.Font("Arial", 14, [System.Drawing.FontStyle]::Bold)
$yellowBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::Yellow)
$g3.DrawString("FORWARD TO ALL GROUPS NOW!!", $fontSub, $yellowBrush, 400, 490, $sf)
$g3.Dispose()
$v3Path = Join-Path $imgDir "v3_whatsapp.png"
Save-BmpAsPng $bmp3 $v3Path
Write-Host "Created $v3Path"

# 4. v4_reddit.png: 800x533, derived from v3, AI tampering on center face (glowing eyes, purple distortion)
Write-Host "Generating v4_reddit.png..."
$bmp4 = New-Object System.Drawing.Bitmap($bmp3)
$g4 = [System.Drawing.Graphics]::FromImage($bmp4)
$g4.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias

# Find center volunteer face coordinates in 800x533 scaled image:
# Original was at ~600,455 in 1200x800 -> in v2 (shifted by 100,66): 500, 389 -> scaled to 800x533: ~400, 311
$fx = 398
$fy = 310

# AI distortion glow / aura
$alienBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(180, 140, 20, 220))
$g4.FillEllipse($alienBrush, $fx - 15, $fy - 12, 50, 48)

# Neon glowing eyes (AI anomaly)
$eyeBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 0, 255, 230))
$g4.FillEllipse($eyeBrush, $fx + 2, $fy + 5, 8, 8)
$g4.FillEllipse($eyeBrush, $fx + 20, $fy + 5, 8, 8)

# Glowing eye beams / artifacts
$beamPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(180, 0, 255, 230), 2)
$g4.DrawLine($beamPen, $fx + 6, $fy + 9, $fx - 25, $fy + 35)
$g4.DrawLine($beamPen, $fx + 24, $fy + 9, $fx + 55, $fy + 35)

$g4.Dispose()
$v4Path = Join-Path $imgDir "v4_reddit.png"
Save-BmpAsPng $bmp4 $v4Path
Write-Host "Created $v4Path"

# 5. v5_instagram.png: 800x800 square crop of v1 with warm vintage filter
Write-Host "Generating v5_instagram.png..."
$bmp5 = New-Object System.Drawing.Bitmap(800, 800)
$g5 = [System.Drawing.Graphics]::FromImage($bmp5)
$g5.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
# Square crop from v1 centered around boat (x: 250, y: 0, w: 800, h: 800)
$srcRect5 = New-Object System.Drawing.Rectangle(250, 0, 800, 800)
$destRect5 = New-Object System.Drawing.Rectangle(0, 0, 800, 800)
$g5.DrawImage($bmp1, $destRect5, $srcRect5, [System.Drawing.GraphicsUnit]::Pixel)

# Warm Instagram filter overlay (Amber/Sepia blend)
$warmOverlay = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(45, 255, 140, 0))
$g5.FillRectangle($warmOverlay, 0, 0, 800, 800)

# Vignette border
$vignettePen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(70, 0, 0, 0), 20)
$g5.DrawRectangle($vignettePen, 10, 10, 780, 780)

$g5.Dispose()
$v5Path = Join-Path $imgDir "v5_instagram.png"
Save-BmpAsPng $bmp5 $v5Path
Write-Host "Created $v5Path"

# 6. v6_news_outlet.png: 750x500 crop of v5 with clean editorial look
Write-Host "Generating v6_news_outlet.png..."
$bmp6 = New-Object System.Drawing.Bitmap(750, 500)
$g6 = [System.Drawing.Graphics]::FromImage($bmp6)
$g6.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$srcRect6 = New-Object System.Drawing.Rectangle(25, 150, 750, 500)
$destRect6 = New-Object System.Drawing.Rectangle(0, 0, 750, 500)
$g6.DrawImage($bmp5, $destRect6, $srcRect6, [System.Drawing.GraphicsUnit]::Pixel)

# Editorial credit in small bottom corner
$fontNews = New-Object System.Drawing.Font("Georgia", 10, [System.Drawing.FontStyle]::Regular)
$newsBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(180, 240, 240, 240))
$g6.DrawString("FOTO: REDES SOCIALES / ARCHIVO", $fontNews, $newsBrush, 15, 475)

$g6.Dispose()
$v6Path = Join-Path $imgDir "v6_news_outlet.png"
Save-BmpAsPng $bmp6 $v6Path
Write-Host "Created $v6Path"

$bmp1.Dispose()
$bmp2.Dispose()
$bmp3.Dispose()
$bmp4.Dispose()
$bmp5.Dispose()
$bmp6.Dispose()

Write-Host "All 6 synthetic test fixtures successfully generated in $imgDir!"
