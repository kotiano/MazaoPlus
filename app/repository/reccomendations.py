def pest_reccomendations():

    PEST_RECOMMENDATIONS = {
        "aphids": (
            "Aphids are tiny sap-sucking bugs that weaken plants and spread viruses.\n"
            "Control methods:\n"
            "- Release ladybugs or lacewings (available from garden stores) to eat aphids naturally.\n"
            "- Spray leaves with soapy water (1 tsp dish soap per liter water) or neem oil weekly.\n"
            "- Blast aphids off plants with a strong hose spray; repeat every 2–3 days.\n"
            "- Rotate crops yearly to prevent aphid buildup.\n"
            "- Plant garlic or marigolds nearby to repel aphids.\n"
            "- Inspect undersides of leaves weekly; squash any aphids found."
        ),
        "bollworm": (
            "Bollworms are worms that chew into buds and fruits, especially cotton and maize.\n"
            "Control methods:\n"
            "- Set up pheromone traps (from farm suppliers) to catch moths before egg-laying.\n"
            "- Use Bt crops (e.g., Bt maize) to kill worms naturally when they feed.\n"
            "- Check plants weekly; remove white egg clusters or worms by hand (wear gloves).\n"
            "- Spray spinosad (natural pesticide) if damage is heavy; follow label instructions.\n"
            "- Plant crops early to avoid peak bollworm season (warm months).\n"
            "- Burn or bury crop residues after harvest to kill hiding worms."
        ),
        "fall armyworm": (
            "Fall armyworms strip leaves and devastate maize, rice, or sorghum.\n"
            "Control methods:\n"
            "- Inspect fields weekly in warm, wet weather; look for young worms or leaf damage.\n"
            "- Release Trichogramma wasps (from suppliers) to destroy armyworm eggs.\n"
            "- Plant Napier grass around fields as a trap crop to lure worms away.\n"
            "- Grow desmodium nearby to repel armyworms with its scent.\n"
            "- Apply Bt spray or Beauveria bassiana (fungal biopesticide) on young worms.\n"
            "- Mow and burn crop debris after harvest to eliminate hiding worms."
        ),
        "stem borer": (
            "Stem borers are larvae that tunnel into stems, weakening or killing plants.\n"
            "Control methods:\n"
            "- Burn or bury crop residues (e.g., maize stalks) after harvest to kill larvae.\n"
            "- Release Cotesia wasps (from suppliers) to attack stem borer larvae.\n"
            "- Plant sorghum as a trap crop near main crops to divert borers.\n"
            "- Spray spinosad when plants are young to protect stems.\n"
            "- Choose borer-resistant crop varieties (check with seed suppliers).\n"
            "- Check stems for holes or frass (sawdust-like waste) weekly; remove larvae."
        ),
        "weevil": (
            "Weevils are beetles that damage grains, fruits, or roots, especially in storage.\n"
            "Control methods:\n"
            "- Remove and burn infested plants or grains to stop weevil spread.\n"
            "- Use sticky traps (from farm stores) to catch adult weevils.\n"
            "- Heat grains to 50°C for 2–3 hours (use solar dryer or oven) to kill eggs.\n"
            "- Mix neem leaves or diatomaceous earth with stored grains to deter weevils.\n"
            "- Store grains in airtight containers to block new weevils.\n"
            "- Check stored grains monthly; discard any with small holes or weevil signs."
        )
    }

    return PEST_RECOMMENDATIONS

def disease_reccomendations():

    DISEASE_RECOMMENDATIONS = {
        "Blight": (
            "Northern Corn Leaf Blight (NCLB), caused by Exserohilum turcicum, forms 1-6 inch cigar-shaped, gray-green to tan lesions on leaves, starting lower.\n"
            "Control methods:\n"
            "- Choose hybrids with partial or race-specific resistance (Ht1, Ht2, or HtN genes).\n"
            "- Scout fields weekly before silking; look for lesions on lower leaves.\n"
            "- Rotate with non-host crops (e.g., wheat) for 1-2 years.\n"
            "- Bury crop debris by plowing.\n"
            "- Plant early to avoid peak humidity periods.\n"
            "- Use fungicides (e.g., Delaro® Complete) if lesions reach the third leaf below the ear on 50% of plants at tasseling."
        ),
        "Common_Rust": (
            "Common Rust, caused by Puccinia sorghi, appears as small, oval, dark-reddish-brown pustules on both leaf surfaces.\n"
            "Control methods:\n"
            "- Plant resistant corn hybrids.\n"
            "- Scout fields weekly from V10-V14; remove affected leaves if limited.\n"
            "- Rotate crops yearly with non-hosts (e.g., soybeans).\n"
            "- Plow crop residues.\n"
            "- Avoid humid areas; ensure good air circulation.\n"
            "- Apply foliar fungicides (e.g., azoxystrobin) if pustules cover 50% of leaves before tasseling."
        ),
        "Gray_Leaf_Spot": (
            "Gray Leaf Spot, caused by Cercospora zeae-maydis, appears as rectangular, grayish-tan lesions with a gray-white center, often along veins.\n"
            "Control methods:\n"
            "- Plant resistant hybrids.\n"
            "- Scout fields weekly during warm, humid conditions (75-85°F).\n"
            "- Rotate with non-host crops for 1-2 years.\n"
            "- Bury crop debris to reduce fungal spores.\n"
            "- Apply fungicides (e.g., strobilurins) at early disease onset, typically at tasseling."
        ),
        "Healthy": (
            "No disease detected. Continue regular monitoring.\n"
            "Recommendations:\n"
            "- Scout fields weekly for early signs of disease.\n"
            "- Maintain crop rotation and soil health.\n"
            "- Ensure proper irrigation."
        )
    }

    return DISEASE_RECOMMENDATIONS