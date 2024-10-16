# `urvc-cli`

CLI for the Ultimate RVC project

**Usage**:

```console
$ urvc-cli [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `song-cover`: Generate song covers

## `urvc-cli song-cover`

Generate song covers

**Usage**:

```console
$ urvc-cli song-cover [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `run-pipeline`: Run the song cover generation pipeline.

### `urvc-cli song-cover run-pipeline`

Run the song cover generation pipeline.

**Usage**:

```console
$ urvc-cli song-cover run-pipeline [OPTIONS] SOURCE MODEL_NAME
```

**Arguments**:

* `SOURCE`: A Youtube URL, the path to a local audio file or the path to a song directory.  [required]
* `MODEL_NAME`: The name of the voice model to use for vocal conversion.  [required]

**Options**:

* `--n-octaves INTEGER`: The number of octaves to pitch-shift the converted vocals by.Use 1 for male-to-female and -1 for vice-versa.  [default: 0]
* `--n-semitones INTEGER`: The number of semi-tones to pitch-shift the converted vocals, instrumentals, and backup vocals by. Altering this slightly reduces sound quality  [default: 0]
* `--f0-method [rmvpe|mangio-crepe]`: The method to use for pitch detection during vocal conversion. Best option is RMVPE (clarity in vocals), then Mangio-Crepe (smoother vocals).  [default: rmvpe]
* `--index-rate FLOAT RANGE`: A decimal number e.g. 0.5, Controls how much of the accent in the voice model to keep in the converted vocals. Increase to bias the conversion towards the accent of the voice model.  [default: 0.5; 0<=x<=1]
* `--filter-radius INTEGER RANGE`: A number between 0 and 7. If >=3: apply median filtering to the pitch results harvested during vocal conversion. Can help reduce breathiness in the converted vocals.  [default: 3; 0<=x<=7]
* `--rms-mix-rate FLOAT RANGE`: A decimal number e.g. 0.25. Controls how much to mimic the loudness of the input vocals (0) or a fixed loudness (1) during vocal conversion.  [default: 0.25; 0<=x<=1]
* `--protect FLOAT RANGE`: A decimal number e.g. 0.33. Controls protection of voiceless consonants and breath sounds during vocal conversion. Decrease to increase protection at the cost of indexing accuracy. Set to 0.5 to disable.  [default: 0.33; 0<=x<=0.5]
* `--hop-length INTEGER`: Controls how often the CREPE-based pitch detection algorithm checks for pitch changes during vocal conversion. Measured in milliseconds. Lower values lead to longer conversion times and a higher risk of voice cracks, but better pitch accuracy. Recommended value: 128.  [default: 128]
* `--room-size FLOAT RANGE`: The room size of the reverb effect applied to the converted vocals. Increase for longer reverb time. Should be a value between 0 and 1.  [default: 0.15; 0<=x<=1]
* `--wet-level FLOAT RANGE`: The loudness of the converted vocals with reverb effect applied. Should be a value between 0 and 1  [default: 0.2; 0<=x<=1]
* `--dry-level FLOAT RANGE`: The loudness of the converted vocals wihout reverb effect applied. Should be a value between 0 and 1.  [default: 0.8; 0<=x<=1]
* `--damping FLOAT RANGE`: The absorption of high frequencies in the reverb effect applied to the converted vocals. Should be a value between 0 and 1.  [default: 0.7; 0<=x<=1]
* `--main-gain INTEGER`: The gain to apply to the post-processed vocals. Measured in dB.  [default: 0]
* `--inst-gain INTEGER`: The gain to apply to the pitch-shifted instrumentals. Measured in dB.  [default: 0]
* `--backup-gain INTEGER`: The gain to apply to the pitch-shifted backup vocals. Measured in dB.  [default: 0]
* `--output-sr INTEGER`: The sample rate of the song cover.  [default: 44100]
* `--output-format [mp3|wav|flac|ogg|m4a|aac]`: The audio format of the song cover.  [default: mp3]
* `--output-name TEXT`: The name of the song cover.
* `--help`: Show this message and exit.
