"""
Astrology Chart Generation Library

This module provides functions to generate Chinese astrology charts using the py_iztro library.
"""

from functools import lru_cache
import json

from py_iztro import Astro


def person_information_extraction(astro_info):
    """Extract basic person information from astro data.
    
    Args:
        astro_info (dict): Astrology information dictionary
        
    Returns:
        str: Formatted person information text
    """
    text_tmp = f"""性别：{astro_info['gender']}
生日：{astro_info['solar_date']}
八字：{astro_info['chinese_date']} {astro_info['time']}"""
    return text_tmp


def extract_major_stars(star_data):
    """Extract major stars information.
    
    Args:
        star_data (dict): Star data for a specific palace
        
    Returns:
        str: Formatted major stars string
    """
    major_stars = star_data['major_stars']
    major = []
    for item in major_stars:
        name = item['name']

        if item['brightness'] != '':
            brightness = '[' + item['brightness'] + ']'
        else:
            brightness = ''

        if item['mutagen'] and item['mutagen'] != '':
            mutagen = f"[生年{item['mutagen']}]"
        else:
            mutagen = ''

        major.append(name + brightness + mutagen)
    return '主星: ' + '+'.join(major)


def extract_minor_stars(star_data):
    """Extract minor stars information.
    
    Args:
        star_data (dict): Star data for a specific palace
        
    Returns:
        str: Formatted minor stars string
    """
    minor_stars = star_data['minor_stars']
    minor = []
    for item in minor_stars:
        name = item['name']

        if item['brightness'] != '':
            brightness = '[' + item['brightness'] + ']'
        else:
            brightness = ''

        minor.append(name + brightness)
    return '辅星: ' + '+'.join(minor)


def extract_adjective_stars(star_data):
    """Extract adjective stars information.
    
    Args:
        star_data (dict): Star data for a specific palace
        
    Returns:
        str: Formatted adjective stars string
    """
    adj_stars = star_data['adjective_stars']
    adj = []
    for item in adj_stars:
        name = item['name']
        adj.append(name)
    return '小星: ' + '+'.join(adj)


def extract_place_info(star_data):
    """Extract place information.
    
    Args:
        star_data (dict): Star data for a specific palace
        
    Returns:
        str: Formatted place information string
    """
    name = star_data['name']
    stem = star_data['heavenly_stem']
    branch = star_data['earthly_branch']
    
    place = name + ('宫' if '命宫' not in name else '') + '[' + stem + branch + ']'
    return place


def extract_big_limit(star_data):
    """Extract big limit (大限) information.
    
    Args:
        star_data (dict): Star data for a specific palace
        
    Returns:
        str: Formatted big limit string
    """
    big_xian_range = star_data['decadal']['range']
    return f"大限:{big_xian_range[0]}~{big_xian_range[1]}虚岁"


def extract_small_limit(star_data):
    """Extract small limit (小限) information.
    
    Args:
        star_data (dict): Star data for a specific palace
        
    Returns:
        str: Formatted small limit string
    """
    small_xian_ages = ','.join(map(str, star_data['ages']))
    return "小限:" + small_xian_ages + "虚岁"


def extract_single_place(star_data):
    """Extract complete information for a single place.
    
    Args:
        star_data (dict): Star data for a specific palace
        
    Returns:
        str: Formatted single place information
    """
    place = extract_place_info(star_data)
    major_star = extract_major_stars(star_data)
    minor_star = extract_minor_stars(star_data)
    adj_star = extract_adjective_stars(star_data)
    big_xian = extract_big_limit(star_data)
    small_xian = extract_small_limit(star_data)

    return f"""├{place}
│├ {major_star}
│├ {minor_star}
│├ {adj_star}
│├ {big_xian}
│└ {small_xian}"""


def generate_full_chart(astrolabe_data):
    """Generate complete astrology chart from astrolabe data.
    
    Args:
        astrolabe_data (list): List of palace data
        
    Returns:
        str: Complete formatted astrology chart
    """
    total_text = []
    
    for star in astrolabe_data:
        text_tmp = extract_single_place(star)
        total_text.append(text_tmp)
        
    return '\n'.join(total_text)


@lru_cache(maxsize=1)
def _get_astro() -> Astro:
    """Create and cache a single Astro engine instance."""
    return Astro()


def full_chart_generation(date, time, gender):
    """Generate complete astrology chart for a person.
    
    Args:
        date (str): Birth date in format "YYYY-M-D"
        time (int): Birth hour (0-23)
        gender (str): Gender ("男" or "女")
        
    Returns:
        str: Complete formatted astrology chart with person information
    """
    astro = _get_astro()
    astrolabe = astro.by_solar(date, time, gender)
    astro_info = json.loads(astrolabe.model_dump_json())
    astro_star_data = astro_info['palaces']

    info_tmp = person_information_extraction(astro_info)
    star_tmp = generate_full_chart(astro_star_data)

    text_tmp = f"""{info_tmp}
命盘：
{star_tmp}"""

    return text_tmp


def get_astro_data(date, time, gender):
    """Get raw astrology data for a person.
    
    Args:
        date (str): Birth date in format "YYYY-M-D"
        time (int): Birth hour (0-23)
        gender (str): Gender ("男" or "女")
        
    Returns:
        dict: Raw astrology data dictionary
    """
    astro = _get_astro()
    astrolabe = astro.by_solar(date, time, gender)
    return json.loads(astrolabe.model_dump_json())
