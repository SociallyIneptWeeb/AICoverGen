# TODO

* test that app works through notebook

* should specify in colab notebook which link in output of last cell should be clicked

* should rename instances of "models" to "voice models"

* have some default models available (ie.e do not need to be downloaded)
  * should be downloaded as part of init.py

## Project/task management

* Should find tool for project/task management
* Tool should support:
  * hierarchical tasks
  * custom labels and or priorities on tasks
  * being able to filter tasks based on those labels
  * being able to close and resolve tasks
  * Being able to integrate with vscode
  * Access for multiple people (in a team)
* Should migrate the content of this file into tool
* Potential candidates
  * GitHub projects
    * Does not yet support hierarchical tasks so no
  * Trello
    * Does not seem to support hierarchical tasks either
  * Notion
    * Seems to support hierarchical tasks, but is complicated
  * Todoist
    * seems to support both hierarchical tasks, custom labels, filtering on those labels, multiple users and there are unofficial plugins for vscode.

## Front end

### Modularization

* Improve modularization of frontend code using helper functions defined [here](https://huggingface.co/spaces/WoWoWoWololo/wrapping-layouts/blob/main/app.py)
* Split front-end modules into further sub-modules.
  * Structure of frontend folder should be:
    * `frontend`
      * `manage_models`
        * `__init__.py`
        * `main.py`
      * `manage_audio`
        * `__init__.py`
        * `main.py`
      * `generate_song_covers`
        * `__init__.py`
        * `main.py`
        * `one_click_generation`
          * `__init__.py`
          * `main.py`
          * `accordions`
            * `__init__.py`
            * `options_x.py` ... ?
        * `multi_step_generation`
          * `__init__.py`
          * `main.py`
          * `accordions`
            * `__init__.py`
            * `step_X.py` ...
      * `common.py`
    * For `multi_step_generation/step_X.py`, its potential render function might have to take the set of all "input tracks" in the multi-step generation tab, so these will then have to be defined in `multi_step_generation/main.py`. Other components passed to `multi_step_generation/main.py` might also need to be passed further down to `multi_step_generation/step_X.py`
    * For `one_click_generation/option_X.py`, its potential render function should
        render the accordion for the given options and return the components defined in the accordion? Other components passed to `one_click_generation/main.py` might also need to be passed further down to `one_click_generation/option_X.py`
  * Import components instead of passing them as inputs to render functions (DIFFICULT TO IMPLEMENT)
    * We have had problems before with component ids when components are instantiated outside a Blocks context in a separate module and then import into other modules and rendered in their blocks contexts.

### Multi-step generation

* Have reset and generate buttons side by side instead of vertically aligned
* Have just one audio separation accordion
* Have just one pitch shift accordion
* Do not auto-transfer output tracks but instead have buttons that allow user to manually transfer output tracks when clicked
* Split components under vocal conversion options into two rows
  * always show "hop length", "detection algorithm" and "protect rate" on second row
* consider making inputs section of each step accordion into a sub-accordion that is closed by default?
* consider having buttons under the "convert vocals" accordion and the pitch shift accordion, which set the pitch shift semi-tones value in the other accordion to be equal to the current value in the current accordion.
* If possible merge two consecutive event listeners using `update_cached_songs` in the song retrieval accordion.
* optimize rendering of audio tracks in ui so that they load quickly (DIFFICULT TO IMPLEMENT)
  * Problem seems to mainly affect step 1: song retrieval and step: 5 vocal post-processing
    * In either case output tracks stop loading after some time
    * `h11._util.LocalProtocolError: Too little data for declared Content-Length` is observed on command line
  * Problem is gradio-specific and an issue has been raised [here](https://github.com/gradio-app/gradio/issues/8878)
* Fix hashes of identical files being different for one-click and multi-step generation (DIFFICULT TO IMPLEMENT)
  * Seems to work except for when first running one click generation and then multi-step generation
  * It is related to the bug mentioned above

### Single step generation

* Split components under vocal conversion options into two rows
  * always show "hop length", "detection algorithm" and "protect rate" on second row

### Common

* save default values for options for song generation in an `SongCoverOptionDefault` enum.
  * then reference this enum across the two tabs
  * and also use `list[SongCoverOptionDefault]` as input to reset settings click event listener in single click generation tab.
* Do not pass dummy checkbox and confirmation state component but instead define them inline in `manage_audio` and `manage_models` frontend modules.
* Try to merge event listeners for confirmation and action execution in `frontend.manage_audio` and `frontend.mange_models`
  * Can output of js function be fed into python function in the same event listener definition?
  * If not possible then try to at least remove dummy input component and dummy/passthrough function in the confirmation event listener
  * See more info here: <https://github.com/gradio-app/gradio/issues/3324>
* get rid of button blocking by setting `concurrency_id = "gpu"` + `concurrency_count=1` for reach event listener calling a function using the gpu
  * Need to figure out which backend functions then actually use GPU
  * In the future we might increase `concurrenct_count` when more GPUs become available
  * NOTE: we still want to continue to block the checkbox for showing intermediate audio files while the one-click generation is running.
* use `Block.queue` with parameter `max_size` set to a non-null value and `default_concurrency_limit` increased in order to improve user responsiveness
* use `Block.launch()` with `max_file_size` to prevent too large uploads
* experiment with `show_error` parameters on `Block.launch()`
* have pop-up success after any kind of job is done ?
  * We can use `gr.Info` for this
  * We already have a helper function `frontend.common.render_msg` which could be used for this purpose (after some small tweaks)
  * In this case we could also consider removing output message textbox components in the `manage_models` and `manage_audio` tabs
* Persist state of app (currently selected settings etc.) across re-renders
  * This includes:
    * refreshing a browser windows
    * Opening app in new browser window
    * Maybe it should also include when app is started anew?
  * Possible solutions
    * Set the `key` attribute of a component when initializing it, so that its state will persist across re-renders
      * Problem is that this solution might not work with accordions or other types of blocks
    * Save any changes to components to a session dictionary and load from it upon refresh
      * See [here](https://github.com/gradio-app/gradio/issues/3106#issuecomment-1694704623)
      * Problem is that this solution might not work with accordions or other types of blocks
    * Use localstorage
      * see [here](https://huggingface.co/spaces/YiXinCoding/gradio-chat-history/blob/main/app.py) and [here](https://huggingface.co/spaces/radames/gradio_window_localStorage/blob/main/app.py)

    * Whenever the state of a component is changed save the new state to a custom JSON file.
      * Then whenever the app is refreshed load the current state of components from the JSON file
      * This solution should probably work for Block types that are not components
* Fix that gradio removes special symbols from audio paths when loaded into audio components (DIFFICULT TO IMPLEMENT)
  * includes parenthesis, question marks, etc.
  * its a gradio bug so report?
* Fix Problem with gradio reload mode not working (DIFFICULT TO IMPLEMENT)
  * Has been reported to gradio here [here](https://github.com/gradio-app/gradio/issues/8917)
* Add button for cancelling any currently running jobs (DIFFICULT TO IMPLEMENT)
  * Not supported by Gradio natively
  * Also difficult to implement manually as Gradio seems to be running called backend functions in thread environments
* dont show error upon missing confirmation (DIFFICULT TO IMPLEMENT)
  * can return `gr.update()`instead of raising an error in relevant event listener function
  * but problem is that subsequent steps will still be executed in this case

### temporary gradio files

* clearing temporary files can be candled with the `delete_cache` parameter
  * Only seems to work if all windows are closed before closing the app process
* When hosting online:
  * clearing of temporary files should happen after a user logs in and out
  * and in this case it should only be temporary files for the active user that are cleared
    * Is that even possible to control?

## Back end

### `common.py`

* Define custom order for items in saved JSON files.
  * Should be the same as the order of items defined in pydantic models?

### `generate_song_cover.py`

* Use a case match expression in `retrieve_song`

```python
case source_type:
    match SourceType.SONG_DIR
        song_path = _get_input_audio_path(song_dir_path)
    match SourceType.URL
    ...
...
```

* Use list comprehension to DRY code in cover pipeline functions
  * Use "double-nested" list comprehension that first makes the necessary calls to `get_unique_base_path` in inner loop and then for each appends `'.wav'` and `'.json'` in outer loop.
  * Apply `all` to result of previous list-comprehension to simplify file path checks
* Make helper function for path validation
  * signature is `validate(files, dirs)`
  * both `files` and `dirs` should have types `list[StrPath]`
  * should check that each item in `files` and `dirs`
    * is not `None`
    * either points to a valid existing file or directory
* Make helper function that both calls `AUDIO_SEPARATOR.separate` and moves its result(s) to the right folder
* split `pitch_shift_background` into
  * one base function `pitch_shift`
  * and simple wrapper function `pitch_shift_background` that calls `pitch_shift`two times
* refactor the three audio separation functions (`separate_vocals`,`separate_main_vocals` and `dereverb`) into
  * One base function `separate_audio`
  * Three wrapper functions (`separate_vocals`,`separate_main_vocals` and `dereverb`) that calls `pitch_shift` with simple parameters
* Implement Pydantic models for the remaining steps in song cover generation pipeline and then instantiate arg dicts by doing `XMetaData.model_dump()`
  * Should we use `model_dump` with `mode = "json"`to ensure JSON-serializable dicts?
* Support specific audio formats for intermediate audio file?
  * We might also want to disable gradio saving input audio to temporary files, as those are always in `.wav` format
    * See more [here](#temporary-gradio-files)
  * Also, it might require some more code to support custom output format for all pipeline functions.

* expand `_get_model_name` so that it can take any audio file in an intermediate audio folder as input (DIFFICULT TO IMPLEMENT)
  * Function should then try to recursively
    * look for a corresponding json metadata file
    * find the model name in that file if it exists
    * otherwise find the path in the input field in the metadata file
    * repeat
  * should also consider whether input audio file belongs to step before audio conversion step
* use pydantic models to constrain numeric inputs (DIFFICULT TO IMPLEMENT)
  * for inputs to `convert` function for example
  * Use `Annotated[basic type, Field[constraint]]` syntax along with a @validate_call decorator on functions
  * Problem is that pyright does not support `Annotated` so we would have to switch to mypy
  
### `manage_models.py`

* use pandas.read_json to load public models table (DIFFICULT TO IMPLEMENT)

## CLI

### Update `display_progress` function

* Always log information message using standard python logging facilities
* rename to `log_progress`?
* use `tqdm` for generating progress bars (PRIORITY ?)
  * by setting the global `gr.Progress` object to track tqdm in the frontend code we will not have to pass this object to the backend at all
  * At the same time we can generate progress bars for CLI as well instead of using `print` to output progress messages
  * have progress bars not be static (one bar shown at a given percentage) but instead show increase in progress up to the given percentage
  * Potential problem: We have to define the tqdm object in each pipeline function
    * This is verbose
    * but more importantly, it might cause problem when we want to have one tqdm bar display when executing the `run_pipeline` function
    * potential solution is to maintain a `progress` parameter for each backend function but instead have it be an instance of a tqdm object
* support a `quiet` mode where no progress bars are displayed?

### Replace `argparse` with `typer` (PRIORITY ?)

* Replace other occurrences of `print` with `typer.echo`
* look at other neat features of `typer`

### Potentially implement audio conversion

* When using CLI input files might not be .wave format. This is a problem because
  * `pedalboard.io` does not support `aac` and `adts` (`m4a`)
    * workaround: accept that these two formats are not supported?
  * `pydub.AudioSegment.from_wav` will not work
    * workaround: use `pydub.AudioSegment.from_file()`?
  * `soundfile` does not support `mp3`,`aac` or `adts`
    * workaround: use `librosa` instead?
* Global solution
  * Implement pre-processing step that saves input file in `.wav` format if necessary before doing anything else and then using that saved audio file for further professing
  * Simpler solution: Always do pre-processing and do not save to file but instead save to a stream that can be used for later processing

### Add remaining CLI interfaces

* Interface for `backend.manage_models`
* Interface for `backend.manage_audio`
* Interfaces for individual pipeline functions defined in `backend.generate_song_covers`

## Scripting

* Fix linting errors in shell script
* Convert batch script to powershell script
* Add timer to  `./urvc install` command
* Update setup scripts so all audio-separation models are downloaded then instead of at runtime
* Synchronize wheel files in `./dependencies/wheels` with upstream repo instead of downloading them from external repo in shell scripts
* Make script for automatic merging of PRs, including:
  * checking out to main after merging
  * pulling latest master
  * deleting local branch
* Update shell scripts so that `./urvc run` and `./urvc dev` both take arbitrary arguments to initialize the Ultimate RVC app with
    1. After pattern matching on the first command to `./urvc` then use `shift` to drop that command and preserve all commands to the right (i.e. shift left)
    2. Then use `$BIN_PATH/python ./src/app.py "$@"`to pass on all the remaining options to the python script

## python package management

* update `requirements.txt`
  * use latest compatible version of all packages
  * remove commented out code, unless strictly necessary
* replace pip with poetry or uv

  * poetry is the standard
  * uv is new and ultra fast (written in rust)

## Audio separation

* expand back-end function(s) so that they are parametrized by both model type as well as model settings
  * Need to decide whether we only want to support common model settings or also settings that are unique to each model
    * It will probably be the latter, which will then require some extra checks.
  * Need to decide which models supported by `audio_separator` that we want to support
    * Not all of them seem to work
    * Probably MDX models and MDXC models
    * Maybe also VR and demucs?
  * Revisit online guide for optimal models and settings
  * Also need to decide whether we want function to consistently return the same type of stem in the same position in return tuple
    * i.e. always return vocal part in first position of tuple
* In multi-step generation tab
  * Expand audio-separation accordion so that model can be selected and appropriate settings for that model can then be selected.
    * Model specific settings should expand based on selected model
* In one-click generation
  * Should have an "vocal extration" option accordion
    * Should be able to choose which audio separation steps to include in pipeline
      * possible steps
        * step 1: separating audio form instrumentals
        * step 2: separating main vocals from background vocals:
        * step 3: de-reverbing vocals
      * Should pick steps from dropdown?
      * For each selected step a new sub-accordion with options for that step will then appear
        * Each accordion should include general settings
        * We should decide whether model specific settings should also be supported
        * We Should also decide whether sub-accordion should setting for choosing a model and if so render specific settings based the chosen model
    * Alternative layout:
      * have option to choose number of separation steps
      * then dynamically render sub accordions for each of the selected number of steps
        * In this case it should be possible to choose models for each accordion
          * this field should be iniitally empty
        * Other setttings should probably have sensible defaults that are the same
      * It might also be a good idea to then have an "examples" pane with recommended combinations of extractions steps
      * When one of these is selected, then the selected number of accordions with the preset settings should be filled out
  * optimize pre-processing
    * check <https://github.com/ArkanDash/Multi-Model-RVC-Inference>
  * Alternatives to `audio-separator` package:
    * [Deezer Spleeter](https://github.com/deezer/spleeter)
      * supports both CLI and python package
    * [Asteroid](https://github.com/asteroid-team/asteroid)
    * [Nuzzle](https://github.com/nussl/nussl)

## GitHub

### Actions

* linting with Ruff
* typechecking with Pyright

### README

* Fill out TBA sections in README
* Add note about not using with VPN?

### Releases

* Make regular releases like done for Applio
  * Will be an `.exe` file that when run unzips contents into application folder, where `./urvc run` can then be executed.
  * Could it be possible to have `.exe` file just start webapp when clicked?

### Other

* In the future consider detaching repo from where it is forked from:
  * because it is not possible to make the repo private otherwise
  * see: <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/detaching-a-fork>

## Incorporate upstream changes

* Incorporate RVC code from [rvc-cli](https://github.com/blaisewf/rvc-cli) (i.e. changes from Applio)
  * more options for voice conversion and more efficient voice conversion
  * batch conversion sub-tab
  * TTS tab
  * Model training tab
  * support more pre-trained models
    * sub-tab under "manage models" tab
  * support for querying online database with many models that can be downloaded
  * support for audio and model analysis.
  * Voice blending tab
* Incorporate latest changes from [RVC-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

## Vocal Conversion

* support arbitrary combination of pitch detection algorithms
  * source: <https://github.com/gitmylo/audio-webui>
* Investigate using onnx models for inference speedup on cpu
* Add more pitch detection methods
  * pm
  * harvest
  * dio
  * rvmpe+
* Implement multi-gpu Inference

## TTS conversion

* also include original edge voice as output
  * source: <https://github.com/litagin02/rvc-tts-webui>

## Model management

### Training models

* have learning rate for training
  * source: <https://github.com/gitmylo/audio-webui>
* have a quick training button
  * or have preprocess dataset, extract features and generate index happen by default
* Support a loss/training graph
  * source: <https://github.com/gitmylo/audio-webui>

### Download models

* Support batch downloading multiple models
  * requires a tabular request form where both a link column and a name column has to be filled out
  * we can allow selecting multiple items from public models table and then copying them over
* support quering online database for models matching a given search string like what is done in applio app
  * first n rows of online database should be shown by default in public models table
    * more rows should be retrieved by scrolling down or clicking a button
  * user search string should filter/narrow returned number of rows in public models table
  * When clicking a set of rows they should then be copied over for downloading in the "download" table
* support a column with preview sample in public models table
  * Only possible if voice snippets are also returned when querying the online database
* Otherwise we can always support voice snippets for voice models that have already been downloaded
  * run model on sample text ("quick brown fox runs over the lazy") after it is downloaded
  * save the results in a `audio/model_preview` folder
  * Preview can then be loaded into a preview audio component when selecting a model from a dropdown
  * or if we replace the dropdown with a table with two columns we can have the audio track displayed in the second column

### Model analysis

* we could provide a new tab to analyze an existing model like what is done in applio
  * or this tab could be consolidated with the delete model tab?

* we could also provide extra model information after model is downloaded
  * potentialy in dropdown to expand?

## Audio management

### General

* Support audio information tool like in applio?
  * A new tab where you can upload a song to analyze?
* more elaborate solution:
  * tab where where you
    * can select any song directory
    * select any step in the audio generation pipeline
    * then select any intermediate audio file generated in that step
    * Then have the possibility to
      * Listen to the song
      * see a table with its metadata (based on its associated `.json` file)
        * add timestamp to json files so they can be sorted in table according to creation date
      * And other statistics in a separate component (graph etc.)
  * Could have delete buttons both at the level of song_directory, step, and for each song?
  * Also consider splitting intermediate audio tracks for each step in to subfolder (0,1,2,3...)

## Audio post-processing

* Support more effects from the `pedalboard` pakcage.
  * Guitar-style effects: Chorus, Distortion, Phaser, Clipping
  * Loudness and dynamic range effects: Compressor, Gain, Limiter
  * Equalizers and filters: HighpassFilter, LadderFilter, LowpassFilter
  * Spatial effects: Convolution, Delay, Reverb
  * Pitch effects: PitchShift
  * Lossy compression: GSMFullRateCompressor, MP3Compressor
  * Quality reduction: Resample, Bitcrush
  * NoiseGate
  * PeakFilter

## Audio Mixing

* Add main gain loudness slider?
* Add option to equalize output audio with respect to input audio
  * i.e. song cover gain (and possibly also more general dynamics) should be the same as those for source song.
  * check to see if pydub has functionality for this
  * otherwise a simple solution would be computing the RMS of the difference between the loudness of the input and output track

  ```python
    rms = np.sqrt(np.mean(np.square(signal)))
    dB  = 20*np.log10(rms)
    #add db to output file in mixing function (using pydub)
  ```

  * When this option is selected the option to set main gain of ouput should be disabled?

* add more equalization options
  * using `pydub.effects` and `pydub.scipy_effects`?

## Custom UI

* Experiment with new themes including [Building new ones](https://www.gradio.app/guides/theming-guid)
  * Support both dark and light theme in app?
  * Add Support for changing theme in app?
  * Use Applio theme as inspiration for default theme?
* Experiment with using custom CSS
  * Pass `css = {css_string}` to `gr.Blocks` and use `elem_classes` and `elem_id` to have components target the styles define in the CSS string.
* Experiment with [custom DataFrame styling](https://www.gradio.app/guides/styling-the-gradio-dataframe)
* Experiment with custom Javascript
* Look for opportunities for defining new useful custom components

## Real-time vocal conversion

* Should support being used as OBS plugin
* Latency is real issue
* Implementations details:
  * implement back-end in Rust?
  * implement front-end using svelte?
  * implement desktop application using C++ or C#?
* see <https://github.com/w-okada/voice-changer> and <https://github.com/RVC-Project/obs-rvc> for inspiration

## AI assistant mode

* similar to vocal conversion streaming but instead of converting your voice on the fly, it should:
  * take your voice,
  * do some language modelling (with an LLM or something)
  * then produce an appropriate verbal response
* We already have Kyutais [moshi](https://moshi.chat/?queue_id=talktomoshi)
  * Maybe that model can be finetuned to reply with a voice
  * i.e. your favorite singer, actor, best friend, family member.

## Ultimate RVC bot for discord

* maybe also make a forum on  discord?

## Make app production ready

* have a "report a bug" tab like in applio?
* should have separate accounts for users when hosting online
  * use `gr.LoginButton` and `gr.LogoutButton`?

* deploy using docker
  * See <https://www.gradio.app/guides/deploying-gradio-with-docker>
* Host on own web-server with Nginx
  * see <https://www.gradio.app/guides/running-gradio-on-your-web-server-with-nginx>

## Colab notebook

* find way of saving virtual environment with python 3.11 in colab notebook (DIFFICULT TO IMPLEMENT)
  * so that this environment can be loaded directly rather than downloading all dependencies every time app is opened

## Testing

* Add example audio files to use for testing
  * Should be located in `audio/examples`
  * could have sub-folders `input` and `output`
    * in `output` folder we have `output_audio.ext` files each with a corresponding `input_audio.json` file containing metadata explaining arguments used to generate output
    * We can then test that actual output is close enough to expected output using audio similarity metric.
* Setup unit testing framework using pytest
