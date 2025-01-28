import pymysql
import logging
from config import MYSQL_CONFIG

def fetch_data_from_mysql():
    try:
        logging.info("Connecting to MySQL database...")
        mysql_conn = pymysql.connect(**MYSQL_CONFIG)
        cursor = mysql_conn.cursor()
        
        query = """SELECT 
                listings.id, 
                listings.name, 
                listings.ref, 
                listings.slug, 
                listings.category, 
                listings.county, 
                listings.county_specific, 
                listings.longitude, 
                listings.latitude, 
                listings.location_description, 
                listings.type as listing_type, 
                listings.class as listing_class, 
                listings.furnishing, 
                listings.bedrooms, 
                listings.bathrooms, 
                listings.sq_area, 
                listings.amount, 
                listings.viewing_fee, 
                listings.property_description, 
                listings.status, 
                listings.availability, 
                listings.subscription_status, 
                listings.complex_id, 
                listings.user_id, 
                listings.created_at, 
                listings.updated_at, 
                listings.link, 
                listings.currency, 
                complexes.title AS complex_title, 
                complexes.ref_code AS complex_ref_code,
                complexes.slug AS complex_slug, 
                complexes.email AS complex_email, 
                complexes.mobile AS complex_mobile, 
                complexes.description AS complex_description, 
                complexes.type AS complex_type, 
                complexes.class AS complex_class, 
                complexes.county AS complex_county, 
                complexes.county_specific AS complex_county_specific, 
                complexes.longitude AS complex_longitude, 
                complexes.latitude AS complex_latitude, 
                complexes.location_description AS complex_location_description, 
                complexes.available AS complex_available,
                GROUP_CONCAT(CONCAT(amenities.type, ': ', amenities.amenity) SEPARATOR '; ') as amenities
            FROM listings 
            LEFT JOIN complexes ON listings.complex_id = complexes.id 
            LEFT JOIN amenities ON listings.id = amenities.listing_id
            WHERE listings.status = 'Published'
            GROUP BY listings.id;"""
        
        cursor.execute(query)
        rows = cursor.fetchall()
        logging.info(f"Fetched {len(rows)} rows from MySQL.")

        # Process rows dynamically instead of assuming a fixed number of values
        for row in rows:
            # Map row to a dictionary with appropriate column names for easy access
            row_dict = {
                'id': row[0],
                'name': row[1],
                'ref': row[2],
                'slug': row[3],
                'category': row[4],
                'county': row[5],
                'county_specific': row[6],
                'longitude': row[7],
                'latitude': row[8],
                'location_description': row[9],
                'listing_type': row[10],
                'listing_class': row[11],
                'furnishing': row[12],
                'bedrooms': row[13],
                'bathrooms': row[14],
                'sq_area': row[15],
                'amount': row[16],
                'viewing_fee': row[17],
                'property_description': row[18],
                'status': row[19],
                'availability': row[20],
                'subscription_status': row[21],
                'complex_id': row[22],
                'user_id': row[23],
                'created_at': row[24],
                'updated_at': row[25],
                'link': row[26],
                'currency': row[27],
                'complex_title': row[28],
                'complex_slug': row[29],
                'complex_email': row[30],
                'complex_mobile': row[31],
                'complex_description': row[32],
                'complex_type': row[33],
                'complex_class': row[34],
                'complex_county': row[35],
                'complex_county_specific': row[36],
                'complex_longitude': row[37],
                'complex_latitude': row[38],
                'complex_location_description': row[39],
                'complex_available': row[40],
            }
            # Do something with the row_dict (e.g., logging, processing, etc.)
            logging.debug(f"Processed row: {row_dict}")

        mysql_conn.close()
        return rows
    except Exception as e:
        logging.error(f"Error fetching data from MySQL: {e}")
        raise


import pymysql
import logging
from config import MYSQL_CONFIG  # Assuming you have this configuration file

def fetch_new_listings_from_mysql(last_processed_id):
    try:
        logging.info("Connecting to MySQL database...")
        mysql_conn = pymysql.connect(**MYSQL_CONFIG)
        cursor = mysql_conn.cursor()
        
        query = """SELECT 
                listings.id, 
                listings.name, 
                listings.ref, 
                listings.slug, 
                listings.category, 
                listings.county, 
                listings.county_specific, 
                listings.longitude, 
                listings.latitude, 
                listings.location_description, 
                listings.type as listing_type, 
                listings.class as listing_class, 
                listings.furnishing, 
                listings.bedrooms, 
                listings.bathrooms, 
                listings.sq_area, 
                listings.amount, 
                listings.viewing_fee, 
                listings.property_description, 
                listings.status, 
                listings.availability, 
                listings.subscription_status, 
                listings.complex_id, 
                listings.user_id, 
                listings.created_at, 
                listings.updated_at, 
                listings.link, 
                listings.currency, 
                complexes.title AS complex_title, 
                complexes.ref_code AS complex_ref_code,
                complexes.slug AS complex_slug, 
                complexes.email AS complex_email, 
                complexes.mobile AS complex_mobile, 
                complexes.description AS complex_description, 
                complexes.type AS complex_type, 
                complexes.class AS complex_class, 
                complexes.county AS complex_county, 
                complexes.county_specific AS complex_county_specific, 
                complexes.longitude AS complex_longitude, 
                complexes.latitude AS complex_latitude, 
                complexes.location_description AS complex_location_description, 
                complexes.available AS complex_available,
                GROUP_CONCAT(CONCAT(amenities.type, ': ', amenities.amenity) SEPARATOR '; ') as amenities
            FROM listings 
            LEFT JOIN complexes ON listings.complex_id = complexes.id 
            LEFT JOIN amenities ON listings.id = amenities.listing_id
            WHERE listings.id > %s AND listings.status = 'Published'
            GROUP BY listings.id;"""
        
        cursor.execute(query, (last_processed_id,))
        new_rows = cursor.fetchall()
        logging.info(f"Fetched {len(new_rows)} new rows from MySQL.")

        # Process rows dynamically instead of assuming a fixed number of values
        for row in new_rows:
            # Map row to a dictionary with appropriate column names for easy access
            row_dict = {
                'id': row[0],
                'name': row[1],
                'ref': row[2],
                'slug': row[3],
                'category': row[4],
                'county': row[5],
                'county_specific': row[6],
                'longitude': row[7],
                'latitude': row[8],
                'location_description': row[9],
                'listing_type': row[10],
                'listing_class': row[11],
                'furnishing': row[12],
                'bedrooms': row[13],
                'bathrooms': row[14],
                'sq_area': row[15],
                'amount': row[16],
                'viewing_fee': row[17],
                'property_description': row[18],
                'status': row[19],
                'availability': row[20],
                'subscription_status': row[21],
                'complex_id': row[22],
                'user_id': row[23],
                'created_at': row[24],
                'updated_at': row[25],
                'link': row[26],
                'currency': row[27],
                'complex_title': row[28],
                'complex_slug': row[29],
                'complex_email': row[30],
                'complex_mobile': row[31],
                'complex_description': row[32],
                'complex_type': row[33],
                'complex_class': row[34],
                'complex_county': row[35],
                'complex_county_specific': row[36],
                'complex_longitude': row[37],
                'complex_latitude': row[38],
                'complex_location_description': row[39],
                'complex_available': row[40],
            }
            # Do something with the row_dict (e.g., logging, processing, etc.)
            logging.debug(f"Processed row: {row_dict}")

        # Update the last processed ID
        if new_rows:
            last_processed_id = max(row[0] for row in new_rows)
        
        mysql_conn.close()
        
        return new_rows, last_processed_id

    except Exception as e:
        logging.error(f"Error fetching new listings from MySQL: {e}")
        raise