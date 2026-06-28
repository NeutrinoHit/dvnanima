# Free Wave Packet Animations

Two lecture-style Manim scenes for scalar wave packets in 2D, visualized as 3D surfaces above the $(x,y)$ plane.

Scenes:

- `SingleGaussianWavePacket`: one moving Gaussian packet with group velocity, spreading, and amplitude decay.
- `TwoFreeWavePackets`: two free packets moving toward each other, interfering during overlap, then recovering and separating again.

Physics:

- the single-packet scene uses the analytic narrow-packet approximation with anisotropic spreading;
- the two-packet scene uses a linear superposition `phi = phi_1 + phi_2` with no interaction.

Files:

- `scene_01_single_packet.py`: scene 1
- `scene_02_two_packets.py`: scene 2
- `numerics.py`: analytic packet formulas and packet kinematics
- `equations.tex`: lecture-ready formulas
- `render_widescreen.sh`: render both scenes in 16:9
- `render_shorts.sh`: render both scenes in 9:16

Shared style:

- `../common/config.py`: profile handling and Manim render geometry
- `../common/camera_presets.py`: camera/layout presets for widescreen and shorts
- `../common/style.py`: palette and surface styling
- `../common/math_text.py`: consistent title/formula panels

Render widescreen:

```bash
cd "/Users/dmitrijnaumov/Library/Mobile Documents/com~apple~CloudDocs/Projects/dvnanima/fields/wavepackets_free"
./render_widescreen.sh
```

Render shorts:

```bash
cd "/Users/dmitrijnaumov/Library/Mobile Documents/com~apple~CloudDocs/Projects/dvnanima/fields/wavepackets_free"
./render_shorts.sh
```

Main parameters:

- packet physics in `numerics.py` (`PacketParameters`)
- scene timing in `scene_01_single_packet.py` and `scene_02_two_packets.py`
- framing and typography in `../common/camera_presets.py`
