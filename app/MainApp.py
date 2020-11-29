from kivy.app import App
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from helper import remove_transparency, val_to_key


class MainApp(App):
    def __init__(self, model, label_dict, target_size, **kwargs):
        # Create a model object
        self.model = model
        self.label_dict = label_dict
        self.target_size = target_size
        # Create a camera object
        self.cameraObject = Camera(play=False)

        # Create a label object
        self.label = Label(text='Prediction: ', font_size='20sp', color=[0, 0, 0])

        super().__init__(**kwargs)

    def build(self):
        full_layout = FloatLayout()

        pred_layout = AnchorLayout(anchor_x='center', anchor_y='top')
        cam_layout = AnchorLayout(anchor_x='center', anchor_y='center')
        but_layout = AnchorLayout(anchor_x='center', anchor_y='bottom')

        self.cameraObject.play = True

        self.cameraObject.resolution = (277, 277)  # Specify the resolution

        # Create a button for taking photograph
        camaraClick = Button(text="Take Photo")

        camaraClick.size_hint = (None, None)

        camaraClick.pos_hint = {'x': .25, 'y': .75}

        # bind the button's on_press to onCameraClick
        camaraClick.bind(on_press=self.onCameraClick)

        # add camera and button to the layout
        pred_layout.add_widget(self.label)
        cam_layout.add_widget(self.cameraObject)
        but_layout.add_widget(camaraClick)

        full_layout.add_widget(pred_layout)
        full_layout.add_widget(cam_layout)
        full_layout.add_widget(but_layout)

        # return the root widget
        return full_layout

    def onCameraClick(self, *args):
        save_dir = 'C:/Users/handw/PycharmProjects/PIKA/app/capture/1.png'
        self.cameraObject.export_to_png(save_dir)
        print("Saved capture image to %s" % save_dir)

        img = remove_transparency(save_dir)

        y_pred = self.model.predict(img.reshape(1, 277, 277, 3))
        y_pred = val_to_key(self.label_dict, y_pred[0].argmax())

        self.label = Label(text='Prediction: %s' % y_pred, font_size='20sp', color=[255, 255, 255])
        print(y_pred)
