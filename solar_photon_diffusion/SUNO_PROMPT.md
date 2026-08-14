# SUNO prompt: Solar photon diffusion

Use **Instrumental** mode. Generate a track at least 65 seconds long; the mix
script trims it to the video and makes a clean four-second fade-out.

## Style prompt (under 1000 characters)

> Instrumental cinematic science-documentary underscore, 60 seconds, 82 BPM,
> spacious and elegant. Begin with a slow warm sub-bass pulse and distant
> glassy harmonics, evoking the dense solar core. Add granular synth particles
> and soft plucks like a microscopic random walk. Around 18 seconds, introduce
> a clear four-step harmonic sequence as the physical model unfolds. From 25 to
> 45 seconds, build gradually with warm analog pads, restrained low strings and
> a steady luminous pulse for slow diffusion through the radiative zone. Near
> 46 seconds, become lighter and more mobile at the convection boundary. End
> with a bright, calm chord near 60 seconds. Awe, precision, patience and
> sunlight; modern documentary mix with space for narration. No vocals, choir,
> lyrics, spoken words, EDM drop, trailer booms, aggressive percussion or
> abrupt ending.

Suggested title: **Forty Thousand Years of Light**.

## File placement and mixing

Download the chosen result as MP3 or WAV and save it as
`music/suno_solar_diffusion.mp3`, then run:

```bash
bash mix_music.sh music/suno_solar_diffusion.mp3
```

This produces both `solar_photon_diffusion_ru_music.mp4` and
`solar_photon_diffusion_en_music.mp4` without re-encoding the video stream.
