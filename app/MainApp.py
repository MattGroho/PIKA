from kivy.clock import Clock
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from kivymd.uix.list import OneLineAvatarListItem, IconLeftWidget
from kivymd.uix.screen import Screen
from kivymd.app import MDApp

from helper import remove_transparency, val_to_key

Window.size = (300, 500)

screen_helper = """
ScreenManager:
    DexScreen:
    CamScreen:
    PokeScreen:

<DexScreen>:
    name: 'dex'
    BoxLayout:
        orientation: 'vertical'
        MDToolbar:
            title: 'PokeDex'
            elevation: 8
        ScrollView:
            MDList:
                id: container
        MDBottomAppBar:
            MDToolbar:
                mode: 'end'
                type: 'bottom'
                icon: 'camera'
                on_action_button: root.manager.current = 'cam'

<CamScreen>:
    name: 'cam'
    BoxLayout:
        orientation: 'vertical'
        Camera:
            id: camera
            resolution: (224, 224)
            play: True
        MDBottomAppBar:
            MDToolbar:
                mode: 'end'
                type: 'bottom'
                icon: 'camera'
                on_action_button: app.capture()
                left_action_items: [['backburger', lambda x: app.change_screen('dex')]]
<PokeScreen>:
    name: 'poke'
    BoxLayout:
        orientation: 'vertical'
        MDFloatLayout:
            Image:
                id: image
                pos_hint: {"x":0, "y":0.25}
                keep_ratio: True
                source: "/Users/handw/PycharmProjects/PIKA/PokeDex/images/bulbasaur.jpg"
            Label:
                id: name
                pos_hint: {"x":0, "y":-0.05}
                text: 'Bulbasaur'
                color: 0,0,0,1
                font_size: '20dp'
        MDBottomAppBar:
            MDToolbar:
                left_action_items: [['backburger', lambda x: app.change_screen('dex')]]
"""


class DexScreen(Screen):
    pass


class CamScreen(Screen):
    pass


class PokeScreen(Screen):
    pass


sm = ScreenManager()
sm.add_widget(DexScreen(name='dex'))
sm.add_widget(CamScreen(name='cam'))
sm.add_widget(PokeScreen(name='poke'))


class PokeList(OneLineAvatarListItem):
    def on_press(self):
        sm.open_poke()


class MainApp(MDApp):
    def __init__(self, model, pokedex, pokedex_dir, label_dict, num_pkmn, target_size, **kwargs):
        # Create a model object
        self.model = model
        self.pokedex = pokedex
        self.pokedex_dir = pokedex_dir
        self.label_dict = label_dict
        self.num_pkmn = num_pkmn
        self.target_size = target_size
        # Create a camera object
        # self.cameraObject = Camera(play=False)

        # Create a label object
        # self.label = Label(text='Prediction: ', font_size='20sp', color=[0, 0, 0])

        super().__init__(**kwargs)

    def build(self):
        # Change theme color to red
        self.theme_cls.primary_palette = 'Red'
        screen = Builder.load_string(screen_helper)

        # return the root widget
        return screen   # DexWidget(self.pokedex, self.pokedex_dir, self.num_pkmn)

    def on_start(self):
        Clock.schedule_once(self.post_init, 0)

    def post_init(self, *args):
        for i in range(0, self.num_pkmn):
            pkmn_name = self.pokedex.getName(i)
            pkmn_file = self.pokedex_dir + pkmn_name + '.jpg'

            # self.pokedex_dir + pkmn_name + '.jpg'
            pkmn_item = OneLineAvatarListItem(text=str(i + 1) + ') ' + pkmn_name)
            pkmn_item.bind(on_release=lambda x, value=(pkmn_name, pkmn_file): self.open_poke(value))
            pkmn_item.add_widget(IconLeftWidget(icon=pkmn_file))
            self.root.get_screen('dex').ids.container.add_widget(pkmn_item)

        self.root.get_screen('cam').ids.camera.resolution = self.target_size

    def open_poke(self, pkmn_info):
        self.root.get_screen('poke').ids.name.text = pkmn_info[0]
        self.root.get_screen('poke').ids.image.source = pkmn_info[1]
        self.change_screen('poke')

    def change_screen(self, screen):
        self.root.current = screen

    def capture(self, *args):
        save_dir = 'C:/Users/handw/PycharmProjects/PIKA/app/capture/1.png'
        self.root.get_screen('cam').ids.camera.export_to_png(save_dir)
        print("Saved capture image to %s" % save_dir)

        img = remove_transparency(save_dir, self.target_size)

        y_pred = self.model.predict(img.reshape(1, self.target_size[0], self.target_size[1], 3))
        pkmn_name = val_to_key(self.label_dict, y_pred[0].argmax())
        #pkmn_name = self.pokedex.getName(y_pred[0].argmax())

        pkmn_file = self.pokedex_dir + pkmn_name + '.jpg'
        self.open_poke((pkmn_name, pkmn_file))
