import json


class PokeDex:
    def __init__(self, lang):

        self.lang = lang
        self.f_names = None
        self.pkmn = None

        self.prep_data()

    def prep_data(self):
        pokedex_dir = "/Users/handw/PycharmProjects/PIKA/PokeDex/"

        # Load folder names
        with open(pokedex_dir + 'f_names.json') as f:
            self.f_names = json.load(f)

        # Load appropriate language file
        with open(pokedex_dir + self.lang + '.json') as f:
            self.pkmn = json.load(f)

    # Return all folder names
    def all_f_names(self):
        return self.f_names

    # Return all pokemon
    def all(self):
        return self.pkmn

    # Get Id of pokemon from name
    def getId(self, name):
        return self.pkmn.index(name)

    # Get name of pokemon from Id
    def getName(self, id):
        return self.pkmn[id]
