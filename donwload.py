from boxmot import BoostTrack  
from pathlib import Path

from torch import device  
  
# This will automatically download to your weights directory  
tracker = BoostTrack(  
    reid_weights=Path('/weights/osnet_x0_25_market1501.pt'),  
    device=device,  
    half=False  
)