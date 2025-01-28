from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import torch
import logging

def generate_narratives_with_gpt(rows):
    try:
        logging.info("Loading GPT model...")
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        narratives = []
        for row in rows:
            (
                id, name, ref, slug, category, county, county_specific, longitude, latitude, location_description,
                listing_type, listing_class, furnishing, bedrooms, bathrooms, sq_area, amount, viewing_fee,
                property_description, status, availability, subscription_status, complex_id, user_id,
                created_at, updated_at, link, currency, 
                complex_title, complex_slug, complex_email, complex_mobile, complex_description, 
                complex_type, complex_class, complex_county, complex_county_specific, complex_longitude, 
                complex_latitude, complex_location_description, complex_available
            ) = row

            # Improved prompt
            prompt = (
                f"Describe a real estate listing based on the following data:\n"
                f"Property Name: {name}\n"
                f"Type: {listing_type}, Class: {listing_class}\n"
                f"Category: {category}, Location: {county_specific}, {county}\n"
                f"Coordinates: Longitude {longitude}, Latitude {latitude}\n"
                f"Price: {currency} {amount}, Viewing Fee: {viewing_fee}\n"
                f"Bedrooms: {bedrooms}, Bathrooms: {bathrooms}, Furnished: {furnishing}, Area: {sq_area} sq. ft.\n"
                f"Complex: {complex_title}, Details: {complex_description}\n"
                f"Focus on highlighting the property details, location, and value. Avoid extra information."
            )

            # Tokenize the prompt
            inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)

            # Generate narrative
            outputs = model.generate(
                inputs,
                max_new_tokens=120,  # Shorten output to stay concise
                temperature=0.5,  # Lower temperature for more deterministic output
                do_sample=False,  # Disable sampling for predictable results
                repetition_penalty=2.5,  # Stronger penalty for repetition
                no_repeat_ngram_size=4,  # Prevent repeated phrases
                pad_token_id=tokenizer.eos_token_id
            )

            # Decode the generated narrative
            narrative = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # Post-process narrative
            if not narrative or "contact" in narrative.lower() or "click" in narrative.lower():
                logging.warning(f"Generated narrative for RefID: {id} is incomplete or irrelevant. Using default message.")
                narrative = "A detailed description of the property is unavailable. Contact for more details."

            narratives.append((id, narrative))

        logging.info(f"Generated narratives for {len(narratives)} rows.")
        return narratives
    except Exception as e:
        logging.error(f"Error generating narratives: {e}")
        raise

def generate_narratives_with_bloom(rows):
    try:
        logging.info("Loading BLOOM model...")
        model_name = "bigscience/bloom"  # You might want to use a specific version based on size or capabilities
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        narratives = []
        for row in rows:
            # Unpack your row data here, same as before

            prompt = (
                f"Generate a precise real estate listing description using only these details:\n"
                f"Property Name: {name}\n"
                f"Type: {listing_type}, Class: {listing_class}, Category: {category}\n"
                f"Location: {county_specific}, {county}\n"
                f"Coordinates: Longitude {longitude}, Latitude {latitude}\n"
                f"Price: {currency} {amount}, Viewing Fee: {viewing_fee}\n"
                f"Bedrooms: {bedrooms}, Bathrooms: {bathrooms}, Furnished: {furnishing}, Area: {sq_area} sq. ft.\n"
                f"Complex: {complex_title}\n"
                f"Create a narrative that highlights the property's features and location benefits. Do not include any external links, contact information, or unrelated content."
            )

            inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            attention_mask = torch.ones(inputs.shape, dtype=torch.long)
            
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.5,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )

            narrative = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = narrative[len(prompt):].strip()
            
            if not generated_part or "contact" in generated_part.lower() or "click" in generated_part.lower():
                logging.warning(f"Generated narrative for RefID: {id} is not suitable or empty. Using default message.")
                generated_part = "Detailed description not available. Please contact for more information."
            
            narratives.append((id, generated_part))

        logging.info(f"Generated narratives for {len(narratives)} rows.")
        return narratives
    except Exception as e:
        logging.error(f"Error generating narratives with BLOOM: {e}")
        raise

def generate_narratives_with_llama(rows):
    try:
        logging.info("Loading Llama 3 model...")
        model_name = "meta-llama/Llama-3-8b"  # Or use "meta-llama/Llama-3-70b" for the bigger model or specify Llama 3.2 if available
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        narratives = []
        for row in rows:
            (
                id, name, ref, slug, category, county, county_specific, longitude, latitude, location_description,
                listing_type, listing_class, furnishing, bedrooms, bathrooms, sq_area, amount, viewing_fee,
                property_description, status, availability, subscription_status, complex_id, user_id,
                created_at, updated_at, link, currency, 
                complex_title, complex_slug, complex_email, complex_mobile, complex_description, 
                complex_type, complex_class, complex_county, complex_county_specific, complex_longitude, 
                complex_latitude, complex_location_description, complex_available
            ) = row

            prompt = (
                f"Generate a precise real estate listing description using only these details:\n"
                f"Property Name: {name}\n"
                f"Type: {listing_type}, Class: {listing_class}, Category: {category}\n"
                f"Location: {county_specific}, {county}\n"
                f"Coordinates: Longitude {longitude}, Latitude {latitude}\n"
                f"Price: {currency} {amount}, Viewing Fee: {viewing_fee}\n"
                f"Bedrooms: {bedrooms}, Bathrooms: {bathrooms}, Furnished: {furnishing}, Area: {sq_area} sq. ft.\n"
                f"Complex: {complex_title}\n"
                f"Create a narrative that highlights the property's features and location benefits. Do not include any external links, contact information, or unrelated content."
            )

            inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            attention_mask = torch.ones(inputs.shape, dtype=torch.long)
            
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.5,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )

            narrative = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = narrative[len(prompt):].strip()
            
            if not generated_part or "contact" in generated_part.lower() or "click" in generated_part.lower():
                logging.warning(f"Generated narrative for RefID: {id} is not suitable or empty. Using default message.")
                generated_part = "Detailed description not available. Please contact for more information."
            
            narratives.append((id, generated_part))

        logging.info(f"Generated narratives for {len(narratives)} rows.")
        return narratives
    except Exception as e:
        logging.error(f"Error generating narratives with Llama 3: {e}")
        raise

def generate_narratives_with_gemma(rows, hf_token):
    try:
        # Log in to Hugging Face
        login(token=hf_token)
        
        logging.info("Loading Gemma 7B model...")
        model_name = "google/gemma-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, torch_dtype=torch.float16)

        narratives = []
        for row in rows:
            (
                id, name, ref, slug, category, county, county_specific, longitude, latitude, location_description,
                listing_type, listing_class, furnishing, bedrooms, bathrooms, sq_area, amount, viewing_fee,
                property_description, status, availability, subscription_status, complex_id, user_id,
                created_at, updated_at, link, currency, 
                complex_title, complex_slug, complex_email, complex_mobile, complex_description, 
                complex_type, complex_class, complex_county, complex_county_specific, complex_longitude, 
                complex_latitude, complex_location_description, complex_available
            ) = row

            prompt = (
                f"Generate a precise real estate listing description using only these details:\n"
                f"Property Name: {name}\n"
                f"Type: {listing_type}, Class: {listing_class}, Category: {category}\n"
                f"Location: {county_specific}, {county}\n"
                f"Coordinates: Longitude {longitude}, Latitude {latitude}\n"
                f"Price: {currency} {amount}, Viewing Fee: {viewing_fee}\n"
                f"Bedrooms: {bedrooms}, Bathrooms: {bathrooms}, Furnished: {furnishing}, Area: {sq_area} sq. ft.\n"
                f"Complex: {complex_title}\n"
                f"Create a narrative that highlights the property's features and location benefits. Do not include any external links, contact information, or unrelated content."
            )
            
            inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            attention_mask = torch.ones(inputs.shape, dtype=torch.long)
            
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.5,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                num_beams=1,  # Beam search can be memory intensive, setting to 1 might help
                early_stopping=False
            )

            narrative = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = narrative[len(prompt):].strip()
            
            if not generated_part or "contact" in generated_part.lower() or "click" in generated_part.lower():
                logging.warning(f"Generated narrative for RefID: {id} is not suitable or empty. Using default message.")
                generated_part = "Detailed description not available. Please contact for more information."
            
            narratives.append((id, generated_part))

        logging.info(f"Generated narratives for {len(narratives)} rows.")
        return narratives
    except Exception as e:
        logging.error(f"Error generating narratives with Gemma 7B: {e}")
        raise

def generate_narrative_manually(rows):
    narratives = []
    logging.info("Loading manual model...")

    for row in rows:
        narrative = []

        (
            id, name, ref, slug, category, county, county_specific, longitude, latitude, location_description,
            listing_type, listing_class, furnishing, bedrooms, bathrooms, sq_area, amount, viewing_fee,
            property_description, status, availability, subscription_status, complex_id, user_id,
            created_at, updated_at, link, currency, 
            complex_title, complex_ref_code, complex_slug, complex_email, complex_mobile, complex_description, 
            complex_type, complex_class, complex_county, complex_county_specific, complex_longitude, 
            complex_latitude, complex_location_description, complex_available, amenities
        ) = row

        # Property Name
        narrative.append(f"Introducing {name}, a {listing_type} located in the vibrant {county_specific}, {county}.")

        # Category
        if category.lower() == 'rent':
            narrative.append(f"This property is available for rent at {currency} {amount} per month.")
        else:
            narrative.append(f"This property is up for sale at {currency} {amount}.")

        # Viewing Fee
        if viewing_fee:
            narrative.append(f"A viewing fee of {viewing_fee} is required to schedule a visit.")

        # Property Details
        if listing_type == 'New Development':
            narrative.append(f"As a new development, {name} offers modern living with state-of-the-art amenities.")
        elif listing_type == 'Apartment':
            narrative.append(f"This {listing_class.lower()} apartment provides urban living at its finest.")
        elif listing_type == 'House':
            narrative.append(f"A {listing_class.lower()} house, {name} promises comfort and space for family living.")
        elif listing_type == 'Cottage':
            narrative.append(f"Experience the charm of a {listing_class.lower()} cottage with {name}.")
        elif listing_type == 'Duplex':
            narrative.append(f"Enjoy the unique layout of this {listing_class.lower()} duplex, {name}.")
        elif listing_type == 'Commercial Unit':
            narrative.append(f"Perfect for business, {name} is a {listing_class.lower()} commercial unit.")
        elif listing_type == 'Penthouse':
            narrative.append(f"Live in luxury with this {listing_class.lower()} penthouse, {name}.")

        # Bedrooms and Bathrooms
        if bedrooms:
            narrative.append(f"It features {bedrooms} {'bedroom' if bedrooms == '1' else 'bedrooms'}.")
        if bathrooms:
            narrative.append(f"And includes {bathrooms} {'bathroom' if bathrooms == 1 else 'bathrooms'}.")

        # Furnishing
        if furnishing:
            narrative.append(f"The property is {furnishing.lower()}.")

        # Square Area
        if sq_area:
            narrative.append(f"With a total area of {sq_area} sq. ft., it offers ample space.")

        # Complex Information
        if complex_id:
            narrative.append(f"{name} is part of the prestigious {complex_title} complex with reference code {complex_ref_code}.")
        else:
            narrative.append(f"{name} stands as an independent property.")

        # Amenities
        if amenities:
            amenity_list = amenities.split('; ')
            for amenity in amenity_list:
                amenity_type, amenity_detail = amenity.split(': ')
                if amenity_type == 'Nearby Amenities':
                    narrative.append(f"Enjoy the convenience of {amenity_detail} just a short distance away.")
                elif amenity_type == 'External Amenities':
                    narrative.append(f"Take advantage of {amenity_detail} right on the property's grounds.")
                elif amenity_type == 'Internal Amenities':
                    narrative.append(f"Inside, you'll find {amenity_detail} for your comfort and enjoyment.")

        # Location Benefits
        narrative.append(f"Located at coordinates Longitude {longitude}, Latitude {latitude}, it's ideally positioned for {'convenience and accessibility' if category.lower() == 'rent' else 'investment and growth'}.")

        # Property Description
        if property_description:
            narrative.append(f"Description: {property_description}")

        # Combine all parts into a single narrative
        full_narrative = ' '.join(narrative)
        
        # Append the generated narrative to the list of narratives
        narratives.append((id, full_narrative))

    logging.info(f"Generated narratives for {len(narratives)} rows.")
    return narratives

def generate_response_with_gpt(query, narratives):
    # Here we're using a simple concatenation of narratives for context, which might need refinement
    context = " ".join(narratives)
    prompt = f"Based on the following property listings:\n{context}\n\nAnswer the query: {query}"
    
    generator = pipeline('text-generation', model='gpt2')
    response = generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
    
    # Clean up the response by removing the prompt
    return response.split(prompt)[-1].strip()