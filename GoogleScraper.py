from selenium import webdriver
import time
import requests
import shutil
import os
import argparse


def save_img(inp, img, i, directory):
    try:
        filename = inp + "_" + str((i-20) + 100) + '.jpg'
        response = requests.get(img, stream=True)
        image_path = os.path.join(directory, filename)
        with open(image_path, 'wb') as file:
            shutil.copyfileobj(response.raw, file)
    except Exception:
        pass


def find_urls(inp, url, driver, directory):
    driver.get(url)
    for _ in range(100):
        driver.execute_script("window.scrollBy(0,10000)")
        try:
            driver.find_element_by_css_selector('.mye4qd').click()
        except:
            continue
    for j, imgurl in enumerate(driver.find_elements_by_xpath('//img[contains(@class,"rg_i Q4LuWd")]')):
        # Cut off bot at over 100 pokemon images
        if j > 120:
            return
        try:
            imgurl.click()
            img = driver.find_element_by_xpath(
                '//body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img').get_attribute(
                "src")
            save_img(inp, img, j, directory)
            time.sleep(0.2)
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape Google images')
    parser.add_argument('-s', '--search', default='bananas', type=str, help='search term')
    parser.add_argument('-d', '--directory', default='/Users/handw/Desktop/testpokemon', type=str, help='save directory')
    args = parser.parse_args()
    driver = webdriver.Chrome('/Users/handw/Desktop/chromedriver.exe')
    directory = args.directory
    if not os.path.isdir(directory):
        os.makedirs(directory)

    pokemon_names = ['Abra', 'Aerodactyl', 'Alakazam', 'Arbok', 'Arcanine', 'Articuno', 'Beedrill', 'Bellsprout', 'Blastoise','Bulbasaur','Butterfree', 'Caterpie', 'Chansey', 'Charizard', 'Charmander', 'Charmeleon', 'Clefable', 'Clefairy', 'Cloyster','Cubone', 'Dewgong', 'Diglett', 'Ditto', 'Dodrio', 'Doduo', 'Dragonair', 'Dragonite', 'Dratini', 'Drowzee', 'Dugtrio', 'Eevee', 'Ekans', 'Electabuzz', 'Electrode', 'Exeggcute', 'Exeggutor', "Farfetch\'d", 'Fearow', 'Flareon', 'Gastly', 'Gengar', 'Geodude', 'Gloom', 'Golbat', 'Goldeen', 'Golduck', 'Golem', 'Graveler', 'Grimer', 'Growlithe', 'Gyarados', 'Haunter', 'Hitmonchan', 'Hitmonlee', 'Horsea', 'Hypno', 'Ivysaur', 'Jigglypuff', 'Jolteon', 'Jynx', 'Kabuto', 'Kabutops', 'Kadabra', 'Kakuna', 'Kangaskhan', 'Kingler', 'Koffing', 'Krabby', 'Lapras', 'Lickitung', 'Machamp', 'Machoke', 'Machop', 'Magikarp', 'Magmar', 'Magnemite', 'Magneton', 'Mankey', 'Marowak', 'Meowth', 'Metapod', 'Mew', 'Mewtwo', 'Moltres', 'Mr.Mime', 'Muk', 'Nidoking', 'Nidoqueen', 'NidoranF', 'NidoranM', 'Nidorina', 'Nidorino', 'Ninetales', 'Oddish', 'Omanyte', 'Omastar', 'Onix', 'Paras', 'Parasect', 'Persian', 'Pidgeot', 'Pidgeotto', 'Pidgey', 'Pikachu', 'Pinsir', 'Poliwag', 'Poliwhirl', 'Poliwrath', 'Ponyta', 'Porygon', 'Primeape', 'Psyduck', 'Raichu', 'Rapidash', 'Raticate','Rattata', 'Rhydon', 'Rhyhorn', 'Sandshrew', 'Sandslash', 'Scyther', 'Seadra', 'Seaking', 'Seel', 'Shellder', 'Slowbro', 'Slowpoke','Snorlax', 'Spearow', 'Squirtle', 'Starmie', 'Staryu', 'Tangela', 'Tauros', 'Tentacool', 'Tentacruel',
     'Vaporeon', 'Venomoth', 'Venonat', 'Venusaur', 'Victreebel', 'Vileplume', 'Voltorb', 'Vulpix', 'Wartortle', 'Weedle', 'Weepinbell', 'Weezing', 'Wigglytuff', 'Zapdos', 'Zubat']

    for pokemon in pokemon_names:
        poke_name = pokemon + " pokemon plush"
        url = 'https://www.google.com/search?q=' + str(
            poke_name) + '&source=lnms&tbm=isch&sa=X&ved=2ahUKEwie44_AnqLpAhUhBWMBHUFGD90Q_AUoAXoECBUQAw&biw=1920&bih=947'
        pokemon_dir = directory + '/' + pokemon

        if not os.path.isdir(pokemon_dir):
            os.makedirs(pokemon_dir)
        find_urls(pokemon, url, driver, pokemon_dir)

        # Take a small break after finishing pokemon folder
        time.sleep(5)
