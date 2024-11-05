
# Gradio notes

## Modularizing large gradio codebases

See this [tutorial](https://www.gradio.app/guides/wrapping-layouts) and corresponding [code](https://huggingface.co/spaces/WoWoWoWololo/wrapping-layouts/blob/main/app.py).

## Event listeners

### Attaching event listeners using decorators

```python
@greet_btn.click(inputs=name, outputs=output)
def greet(name):
    return "Hello " + name + "!"
```

### Function input using dicts

```python
a = gr.Number(label="a")
b = gr.Number(label="b")

def sub(data):
    return data[a] - data[b]
sub_btn.click(sub, inputs={a, b}, outputs=c)
```

This syntax may be better for functions with many inputs

### Function output using dicts

```python
food_box = gr.Number(value=10, label="Food Count")
status_box = gr.Textbox()

def eat(food):
    if food > 0:
        return {food_box: food - 1, status_box: "full"}
    else:
        return {status_box: "hungry"}

gr.Button("Eat").click(
    fn=eat,
    inputs=food_box,
    outputs=[food_box, status_box]
)
```

Allows you to skip updating some output components.

### Binding multiple event listeners to one function

```python
name = gr.Textbox(label="Name")
output = gr.Textbox(label="Output Box")
greet_btn = gr.Button("Greet")
trigger = gr.Textbox(label="Trigger Box")

def greet(name, evt_data: gr.EventData):
    return "Hello " + name + "!", evt_data.target.__class__.__name__

def clear_name(evt_data: gr.EventData):
    return ""

gr.on(
    triggers=[name.submit, greet_btn.click],
    fn=greet,
    inputs=name,
    outputs=[output, trigger],
).then(clear_name, outputs=[name])
```

* Use `gr.on` with optional `triggers` argument. If `triggers` is not set then the given function will be called for all `.change` event listeners in the app.
* Allows you to DRY a lot of code potentially.

### Running events continuously

```python
with gr.Blocks as demo:
    timer = gr.Timer(5)
    textbox = gr.Textbox()
    textbox2 = gr.Textbox()
    timer.tick(set_textbox_fn, textbox, textbox2)
```

Or alternatively the following semantics can be used:

```python
with gr.Blocks as demo:
timer = gr.Timer(5)
textbox = gr.Textbox()
textbox2 = gr.Textbox(set_textbox_fn, inputs=[textbox], every=timer)
```

## Other semantics

### Conditional component values

```python
with gr.Blocks() as demo:
    num1 = gr.Number()
    num2 = gr.Number()
    product = gr.Number(lambda a, b: a * b, inputs=[num1, num2])
```

* Value of component must be a function taking two component values and returning a new component value
* Component must also take a list of inputs indicating which other components should be used to compute its value
* Components value will always be updated whenever the other components `.change` event listeners are called.
* Hence this method can be used to DRY code with many `.change` event listeners

### Dynamic behavior

We can use the `@gr.render` decorator to dynamically define components and event listeners while an app is executing

#### Dynamic components

```python
import gradio as gr

    with gr.Blocks() as demo:
        input_text = gr.Textbox(label="input")

        @gr.render(inputs=input_text)
        def show_split(text):
            if len(text) == 0:
                gr.Markdown("## No Input Provided")
            else:
                for letter in text:
                    gr.Textbox(letter)

    demo.launch()
```

By default `@gr.render` is called whenever the `.change` event for the given input components are executed or when the app is loaded. This can be overriden by also giving a triggers argument to the decorator:

```python
@gr.render(inputs=input_text, triggers = [input_text.submit])
...
```

#### Dynamic event listeners

```python
with gr.Blocks() as demo:
    text_count = gr.State(1)
    add_btn = gr.Button("Add Box")
    add_btn.click(lambda x: x + 1, text_count, text_count)

    @gr.render(inputs=text_count)
    def render_count(count):
        boxes = []
        for i in range(count):
            box = gr.Textbox(key=i, label=f"Box {i}")
            boxes.append(box)

        def merge(*args):
            return " ".join(args)

        merge_btn.click(merge, boxes, output)

    merge_btn = gr.Button("Merge")
    output = gr.Textbox(label="Merged Output")
```

* All event listeners that use components created inside a render function must also be defined inside that render function
* The event listener can still reference components outside the render function
* Just as with components, whenever a function re-renders, the event listeners created from the previous render are cleared and the new event listeners from the latest run are attached.
* setting `key = ...` when instantiating a  component ensures that the value of the component is preserved upon rerender
  * This is might also allow us to preserve session state easily across browser refresh?

#### A more elaborate example

```python
import gradio as gr

with gr.Blocks() as demo:

    tasks = gr.State([])
    new_task = gr.Textbox(label="Task Name", autofocus=True)

    def add_task(tasks, new_task_name):
        return tasks + [{"name": new_task_name, "complete": False}], ""

    new_task.submit(add_task, [tasks, new_task], [tasks, new_task])

    @gr.render(inputs=tasks)
    def render_todos(task_list):
        complete = [task for task in task_list if task["complete"]]
        incomplete = [task for task in task_list if not task["complete"]]
        gr.Markdown(f"### Incomplete Tasks ({len(incomplete)})")
        for task in incomplete:
            with gr.Row():
                gr.Textbox(task['name'], show_label=False, container=False)
                done_btn = gr.Button("Done", scale=0)
                def mark_done(task=task):
                    task["complete"] = True
                    return task_list
                done_btn.click(mark_done, None, [tasks])

                delete_btn = gr.Button("Delete", scale=0, variant="stop")
                def delete(task=task):
                    task_list.remove(task)
                    return task_list
                delete_btn.click(delete, None, [tasks])

        gr.Markdown(f"### Complete Tasks ({len(complete)})")
        for task in complete:
            gr.Textbox(task['name'], show_label=False, container=False)

demo.launch()
```

* Any event listener that modifies a state variable in a manner that should trigger a re-render must set the state variable as an output. This lets Gradio know to check if the variable has changed behind the scenes.
* In a `gr.render`, if a variable in a loop is used inside an event listener function, that variable should be "frozen" via setting it to itself as a default argument in the function header. See how we have task=task in both mark_done and delete. This freezes the variable to its "loop-time" value.

### Progress bars

Instead of doing `gr.progress(percentage, desc= "...")` in core helper functions you can just use tqdm directly in your code by instantiating `gr.progress(track_tqdm = true)` in a web helper function/harness.

Alternatively, you can also do `gr.Progress().tqdm(iterable, description, total, unit)` to attach a tqdm iterable to the progress bar

Benefits of either approach is:

* we do not have to supply a `gr.Progress` object to core functions.
* Perhaps it will also be possible to get a progress bar that automatically generates several update steps for a given caption, rather than just one step as is the case when using `gr.Progress`

### State

Any variable created outside a function call is shared by all users of app

So when deploying app in future need to use `gr.State()` for all variables declared outside functions?

## Notes on Gradio classes

* `Blocks.launch()`
  * `prevent_thread_lock` can be used to have an easier way of shutting down app?
  * `show_error`: if `True`can allow us not to have to reraise core exceptions as `gr.Error`?
* `Tab`
  * event listener triggered when tab is selected could be useful?
* `File`
  * `file_type`: can use this to limit input types to .pth, .index and .zip when downloading a model
* `Label`
  * Intended for output of classification models
  * for actual labels in UI maybe use `gr.Markdown`?

* `Button`
  * `link`: link to open when button is clicked?
  * `icon`: path to icon to display on button

* `Audio`: relevant event listeners:
  * `upload`: when a value is uploaded
  * `input`: when a value is changed
  * `clear`: when a value is cleared
* `Dropdown`
  * `height`
  * `min_width`
  * `wrap`: if text in cells should wrap
  * `column_widths`: width of each column
  * `datatype`: list of `"str"`, `"number"`, `"bool"`, `"date"`, `"markdown"`

## Performance optimization

* Can set `max_threads` argument for `Block.launch()`
if you have any async definitions in your code (`async def`).
* can set `max_size` argument on `Block.queue()`. This limits how many people can wait in line in the queue. If too many people are in line, new people trying to join will receive an error message. This can be better than default which is just having people wait indefinitely
* Can increase `default_concurrency_limit` for `Block.queue()`. Default is `1`. Increasing to more might make operations more effective.
* Rewrite functions so that they take a batched input and set `batched = True` on the event listener calling the function

## Environment Variables

Gradio supports environment variables which can be used to customize the behavior
of your app from the command line instead of setting these parameters in `Blocks.launch()`

* GRADIO_ANALYTICS_ENABLED
* GRADIO_SERVER_PORT
* GRADIO_SERVER_NAME
* GRADIO_TEMP_DIR
* GRADIO_SHARE
* GRADIO_ALLOWED_PATHS
* GRADIO_BLOCKED_PATHS

These could be useful when running gradio apps from a shell script.

## Networking

### File Access

Users can access:

* Temporary files created by gradio
* Files that are allowed via the `allowed_paths` parameter set in `Block.launch()`
* static files that are set via [gr.set_static_paths](https://www.gradio.app/docs/gradio/set_static_paths)
  * Accepts a list of directories or files names that will not be copied to the cached but served directly from computer.
  * BONUS: This can be used in ULTIMATE RVC for dispensing with the temp gradio directory. Need to consider possible ramifications before implementing this though.

Users cannot access:

* Files that are blocked via the `blocked_paths` parameter set in `Block.launch()`
  * This parameter takes precedence over the `allowed_paths` parameter and over default allowed paths
* Any other paths on the host machine
  * This is something to consider when hosting app online

#### Limiting file upload size

you can use `Block.launch(max_file_size= ...)` to limit max file size in MBs for each user.

### Access network request

you can access information from a network request directly within a gradio app:

```python
import gradio as gr

def echo(text, request: gr.Request):
    if request:
        print("Request headers dictionary:", request.headers)
        print("IP address:", request.client.host)
        print("Query parameters:", dict(request.query_params))
    return text

io = gr.Interface(echo, "textbox", "textbox").launch()
```

If the network request is not done via the gradio UI then it will be `None` so always check if it exists

### Authentication

#### Password protection

You can have an authentication page in front of your app by doing:

```python
demo.launch(auth=("admin", "pass1234"))
```

More complex handling can be achieved by giving a function as input:

```python
def same_auth(username, password):
    return username == password
demo.launch(auth=same_auth)
```

Also support a logout page:

```python
import gradio as gr

def update_message(request: gr.Request):
    return f"Welcome, {request.username}"

with gr.Blocks() as demo:
    m = gr.Markdown()
    logout_button = gr.Button("Logout", link="/logout")
    demo.load(update_message, None, m)
    
demo.launch(auth=[("Pete", "Pete"), ("Dawood", "Dawood")])
```

NOTE:

* For authentication to work properly, third party cookies must be enabled in your browser. This is not the case by default for Safari or for Chrome Incognito Mode.
* Gradio's built-in authentication provides a straightforward and basic layer of access control but does not offer robust security features for applications that require stringent access controls (e.g. multi-factor authentication, rate limiting, or automatic lockout policies).

##### Custom user content

Customize content for each user by accessing the network request directly:

```python
import gradio as gr

def update_message(request: gr.Request):
    return f"Welcome, {request.username}"

with gr.Blocks() as demo:
    m = gr.Markdown()
    demo.load(update_message, None, m)
    
demo.launch(auth=[("Abubakar", "Abubakar"), ("Ali", "Ali")])
```

#### OAuth Authentication

See <https://www.gradio.app/guides/sharing-your-app#o-auth-with-external-providers>

## Styling

### UI Layout

#### `gr.Row`

* `equal_height = false` will not force component on the same row to have the same height
* experiment with `variant = 'panel'` or `variant = 'compact'` for different look

#### `gr.Column`

* experiment with `variant = 'panel'` or `variant = 'compact'` for different look

#### `gr.Block`

* `fill_height = True` and `fill_width = True` can be used to fill browser window

#### `gr.Component`

* `scale = 0` can be used to prevent component from expanding to take up space.

### DataFrame styling

See <https://www.gradio.app/guides/styling-the-gradio-dataframe>

### Themes

```python
with gr.Blocks(theme=gr.themes.Glass()):
...
```

See this [theming guide](https://www.gradio.app/guides/theming-guide) for how to create new custom themes both using the gradio theme builder

### Custom CSS

Change background color to red:

```python
with gr.Blocks(css=".gradio-container {background-color: red}") as demo:
...
```

Set background to image file:

```python
with gr.Blocks(css=".gradio-container {background: url('file=clouds.jpg')}") as demo:
...
```

#### Customize Component style

Use `elem_id` and `elem_classes` when instantiating component. This will allow you to select elements more easily with CSS:

```python
css = """
#warning {background-color: #FFCCCB}
.feedback textarea {font-size: 24px !important}
"""

with gr.Blocks(css=css) as demo:
    box1 = gr.Textbox(value="Good Job", elem_classes="feedback")
    box2 = gr.Textbox(value="Failure", elem_id="warning", elem_classes="feedback")
```

* `elem_id` adds an HTML element id to the specific component
* `elem_classes`adds a class or list of classes to the component.

## Custom front-end logic

### Custom Javascript

You can add javascript

* as a string or file path when instantiating a block:
```blocks(js = path or string)```
  * Javascript will be executed when app loads?
* as a string to an event listener. This javascript code will be executed before the main function attached to the event listner.
* add javascript code to the head param of the blocks initializer. This will add the code to the head of the HTML document:

    ```python
    head = f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={google_analytics_tracking_id}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', '{google_analytics_tracking_id}');
    </script>
    """

    with gr.Blocks(head=head) as demo:
        ...demo code...
    ```

### Custom Components

See <https://www.gradio.app/guides/custom-components-in-five-minutes>

## Connecting to databases

Might be useful when we need to retrieve voice models hosted online later.

Can import data using a combination of `sqlalchemy.create_engine` and `pandas.read_sql_query`:

```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('sqlite:///your_database.db')

with gr.Blocks() as demo:
    origin = gr.Dropdown(["DFW", "DAL", "HOU"], value="DFW", label="Origin")

    gr.LinePlot(
        lambda origin: pd.read_sql_query(
            f"SELECT time, price from flight_info WHERE origin = {origin};", 
            engine
        ), inputs=origin, x="time", y="price")
```

## Sharing a Gradio App

### Direct sharing

* You can do `Blocks.launch(share = True)` to launch app on a public link that expires in 72 hours
* IT is possible to set up your own Share Server on your own cloud server to overcome this restriction
  * See <https://github.com/huggingface/frp/>

### Embedding hosted HF space

You can embed a gradio app hosted on huggingface spaces into any other web app.

## Gradio app in production

Useful information for migrating gradio app to production.

### App hosting

#### Custom web-server with Nginx

see <https://www.gradio.app/guides/running-gradio-on-your-web-server-with-nginx>

#### Deploying a gradio app with docker

See <https://www.gradio.app/guides/deploying-gradio-with-docker>

#### Running serverless apps

Web apps hosted completely in your browser (without any server for backend) can be implemented using a combination of Gradio lite + transformers.js.

More information:

* <https://www.gradio.app/guides/gradio-lite>
* <https://www.gradio.app/guides/gradio-lite-and-transformers-js>

#### Zero-GPU spaces

In development.

see <https://www.gradio.app/main/docs/python-client/using-zero-gpu-spaces>

#### Analytics dashboard

Used for monitoring traffic.

Analytics can be disabled by setting `analytics_enabled = False` as argument to `gr.Blocks()`

### Gradio App as API

Each gradio app has a button that redirects you to documentation for a corresponding API. This API can be called via:

* Dedicated [Python](https://www.gradio.app/guides/getting-started-with-the-python-client) or [Javascript](https://www.gradio.app/guides/getting-started-with-the-js-client) API clients.
* [Curl](https://www.gradio.app/guides/querying-gradio-apps-with-curl)
* Community made [Rust client](https://www.gradio.app/docs/third-party-clients/rust-client).

Alternatively, one can

* mount gradio app within existing fastapi application
* do a combination where the python gradio client is used inside fastapi app to query an endpoint from a gradio app.

#### Mounting app within FastAPI app

```python
from fastapi import FastAPI
import gradio as gr

CUSTOM_PATH = "/gradio"

app = FastAPI()

@app.get("/")
def read_main():
    return {"message": "This is your main app"}

io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
app = gr.mount_gradio_app(app, io, path=CUSTOM_PATH)
```

* Run this from the terminal as you would normally start a FastAPI app: `uvicorn run:app`
* and navigate to <http://localhost:8000/gradio> in your browser.

#### Using a block context as a function to call

```python
english_translator = gr.load(name="spaces/gradio/english_translator")
def generate_text(text):
    english_text = english_generator(text)[0]["generated_text"]
```

If the app you are loading defines more than one function, you can specify which function to use with the `fn_index` and `api_name` parameters:

```python
translate_btn.click(translate, inputs=english, outputs=german, api_name="translate-to-german")
....
english_generator(text, api_name="translate-to-german")[0]["generated_text"]
```

#### Automatic API documentation

1. Record api calls to generate snippets of calls made in app. Gradio

2. Gradio can then reconstruct documentation describing what happened

#### LLM agents

LLM agents such as those defined using LangChain can call gradio apps and compose the results they produce.

More information: <https://www.gradio.app/guides/gradio-and-llm-agents>
