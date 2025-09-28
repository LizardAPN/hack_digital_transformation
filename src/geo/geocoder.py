import requests
import logging
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)


class Geocoder:
    """Класс для работы с геокодерами и получения адресов по координатам"""
    
    def __init__(self):
        """Инициализация геокодера"""
        self.yandex_api_key = os.getenv("YANDEX_GEOCODER_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_GEOCODER_API_KEY")
        self.openstreetmap_enabled = True
        
    def geocode(self, lat: float, lon: float) -> Optional[str]:
        """
        Получение адреса по координатам с использованием различных геокодеров
        
        Args:
            lat: Широта
            lon: Долгота
            
        Returns:
            Адрес в виде строки или None
        """
        # Пробуем различные геокодеры в порядке приоритета
        geocoders = [
            self._yandex_geocode,
            self._google_geocode,
            self._openstreetmap_geocode
        ]
        
        for geocoder_func in geocoders:
            try:
                address = geocoder_func(lat, lon)
                if address:
                    return address
            except Exception as e:
                logger.warning(f"Ошибка геокодера {geocoder_func.__name__}: {e}")
                continue
                
        return None
    
    def _yandex_geocode(self, lat: float, lon: float) -> Optional[str]:
        """
        Геокодирование с помощью Yandex Geocoder API
        
        Args:
            lat: Широта
            lon: Долгота
            
        Returns:
            Адрес в виде строки или None
        """
        if not self.yandex_api_key:
            logger.warning("Yandex API key не установлен")
            return None
            
        try:
            url = "https://geocode-maps.yandex.ru/1.x/"
            params = {
                "apikey": self.yandex_api_key,
                "geocode": f"{lon},{lat}",
                "format": "json",
                "kind": "house",
                "results": 1
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            # Извлекаем адрес из ответа
            feature_member = data.get("response", {}).get("GeoObjectCollection", {}).get("featureMember", [])
            if feature_member:
                address = feature_member[0].get("GeoObject", {}).get("metaDataProperty", {}).get("GeocoderMetaData", {}).get("text")
                if address:
                    return address
                    
            return None
        except Exception as e:
            logger.error(f"Ошибка Yandex геокодера: {e}")
            return None
    
    def _google_geocode(self, lat: float, lon: float) -> Optional[str]:
        """
        Геокодирование с помощью Google Geocoding API
        
        Args:
            lat: Широта
            lon: Долгота
            
        Returns:
            Адрес в виде строки или None
        """
        if not self.google_api_key:
            logger.warning("Google API key не установлен")
            return None
            
        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                "latlng": f"{lat},{lon}",
                "key": self.google_api_key,
                "language": "ru"
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            # Извлекаем адрес из ответа
            results = data.get("results", [])
            if results:
                address = results[0].get("formatted_address")
                if address:
                    return address
                    
            return None
        except Exception as e:
            logger.error(f"Ошибка Google геокодера: {e}")
            return None
    
    def _openstreetmap_geocode(self, lat: float, lon: float) -> Optional[str]:
        """
        Геокодирование с помощью OpenStreetMap Nominatim
        
        Args:
            lat: Широта
            lon: Долгота
            
        Returns:
            Адрес в виде строки или None
        """
        if not self.openstreetmap_enabled:
            return None
            
        try:
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                "lat": lat,
                "lon": lon,
                "format": "json",
                "addressdetails": 1,
                "accept-language": "ru"
            }
            
            headers = {
                "User-Agent": "BuildingDetector/1.0 (hackathon project)"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            # Извлекаем адрес из ответа
            address = data.get("display_name")
            if address:
                return address
                
            return None
        except Exception as e:
            logger.error(f"Ошибка OpenStreetMap геокодера: {e}")
            return None


# Глобальный экземпляр геокодера
geocoder_instance = None


def get_geocoder() -> Geocoder:
    """Получение глобального экземпляра геокодера"""
    global geocoder_instance
    if geocoder_instance is None:
        geocoder_instance = Geocoder()
    return geocoder_instance


def geocode_coordinates(lat: float, lon: float) -> Optional[str]:
    """
    Получение адреса по координатам
    
    Args:
        lat: Широта
        lon: Долгота
        
    Returns:
        Адрес в виде строки или None
    """
    geocoder = get_geocoder()
    return geocoder.geocode(lat, lon)
