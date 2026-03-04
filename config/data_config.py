"""
Configuration for the data.
"""

DAILY = "TIME_SERIES_DAILY"
DAILY_ADJUSTED = "TIME_SERIES_DAILY_ADJUSTED"
WINDOW_START = "2000-01-01"
WINDOW_END = "2019-12-31"
TICKERS = [
    "NVDA",   # NVIDIA
    "AAPL",   # Apple
    "GOOG",   # Alphabet (Google)
    "MSFT",   # Microsoft
    "AMZN",   # Amazon
    "META",   # Meta Platforms
    "AVGO",   # Broadcom
    "TSLA",   # Tesla
    "BRK-B",  # Berkshire Hathaway
    "WMT",    # Walmart
    "LLY",    # Eli Lilly
    "JPM",    # JPMorgan Chase
    "XOM",    # Exxon Mobil
    "JNJ",    # Johnson & Johnson
    "V",      # Visa
    "MU",     # Micron Technology
    "MA",     # Mastercard
    "COST",   # Costco
    "ORCL",   # Oracle
    "ABBV",   # AbbVie
    "PG",     # Procter & Gamble
    "HD",     # Home Depot
    "CVX",    # Chevron
    "BAC",    # Bank of America
    "GE",     # General Electric
    "CAT",    # Caterpillar
    "AMD",    # Advanced Micro Devices
    "KO",     # Coca-Cola
    "NFLX",   # Netflix
    "MRK",    # Merck
    "CSCO",   # Cisco
    "PLTR",   # Palantir
    "LRCX",   # Lam Research
    "AMAT",   # Applied Materials
    "PM",     # Philip Morris International
    "GS",     # Goldman Sachs
    "MS",     # Morgan Stanley
    "RTX",    # RTX
    "WFC",    # Wells Fargo
    "TMUS",   # T-Mobile US
    "UNH",    # UnitedHealth
    "GEV",    # GE Vernova
    "MCD",    # McDonald's
    "PEP",    # PepsiCo
    "INTC",   # Intel
    "AXP",    # American Express
    "IBM",    # IBM
    "VZ",     # Verizon
    "AMGN",   # Amgen
    "T",      # AT&T
    "ABT",    # Abbott Laboratories
    "NEE",    # Nextera Energy
    "KLAC",   # KLA
    "C",      # Citigroup
    "TXN",    # Texas Instruments
    "TMO",    # Thermo Fisher Scientific
    "DIS",    # Walt Disney
    "APH",    # Amphenol
    "BA",     # Boeing
    "GILD",   # Gilead Sciences
    "CRM",    # Salesforce
    "TJX",    # TJX Companies
    "ISRG",   # Intuitive Surgical
    "DE",     # Deere & Company
    "ADI",    # Analog Devices
    "SCCO",   # Southern Copper
    "SCHW",   # Charles Schwab
    "BLK",    # BlackRock
    "HON",    # Honeywell
    "ANET",   # Arista Networks
    "UNP",    # Union Pacific
    "LOW",    # Lowe's
    "QCOM",   # Qualcomm
    "PFE",    # Pfizer
    "LMT",    # Lockheed Martin
    "UBER",   # Uber
    "DHR",    # Danaher
    "WELL",   # Welltower
    "SYK",    # Stryker
    "BX",     # Blackstone
    "NEM",    # Newmont
    "COP",    # ConocoPhillips
    "APP",    # AppLovin
    "BKNG",   # Booking Holdings
    "PLD",    # Prologis
    "GLW",    # Corning
    "PH",     # Parker-Hannifin
    "SPGI",   # S&P Global
    "BMY",    # Bristol-Myers Squibb
    "COF",    # Capital One
    "VRTX",   # Vertex Pharmaceuticals
    "IBKR",   # Interactive Brokers
    "PGR",    # Progressive
    "MCK",    # McKesson
    "HCA",    # HCA Healthcare
    "MO",     # Altria Group
    "CMCSA",  # Comcast
    "CME",    # CME Group
    "PANW",   # Palo Alto Networks
    "CEG",    # Constellation Energy
]
TICKERS2 = [
    "TSM",    # Taiwan Semiconductor
    "ASML",   # ASML Holding
    "TM",     # Toyota Motor
    "BABA",   # Alibaba
    "NVS",    # Novartis
    "AZN",    # AstraZeneca
    "HSBC",   # HSBC Holdings
    "SHEL",   # Shell
    "LIN",    # Linde
    "RY",     # Royal Bank of Canada
    "SAP",    # SAP
    "BHP",    # BHP Group
    "MUFG",   # Mitsubishi UFJ Financial
    "SAN",    # Banco Santander
    "TTE",    # TotalEnergies
    "RIO",    # Rio Tinto
    "NVO",    # Novo Nordisk
    "TD",     # Toronto-Dominion Bank
    "UL",     # Unilever
    "SHOP",   # Shopify
    "BUD",    # Anheuser-Busch InBev
    "HDB",    # HDFC Bank
    "ETN",    # Eaton
    "PDD",    # PDD Holdings
    "SMFG",   # Sumitomo Mitsui Financial
    "SONY",   # Sony Group
    "BTI",    # British American Tobacco
    "CB",     # Chubb
    "ARM",    # Arm Holdings
    "ACN",    # Accenture
    "AEM",    # Agnico Eagle Mines
    "UBS",    # UBS Group
    "MDT",    # Medtronic
    "BBVA",   # Banco Bilbao Vizcaya Argentaria
    "ENB",    # Enbridge
    "GSK",    # GSK
    "INTU",   # Intuit
    "SNY",    # Sanofi
    "NOW",    # ServiceNow
    "BSX",    # Boston Scientific
    "SBUX",   # Starbucks
    "NOC",    # Northrop Grumman
    "SO",     # Southern Company
    "IBN",    # ICICI Bank
    "PBR",    # Petrobras
    "ADBE",   # Adobe
    "HWM",    # Howmet Aerospace
    "MFG",    # Mizuho Financial
    "SPOT",   # Spotify
    "CVS",    # CVS Health
    "TT",     # Trane Technologies
    "BMO",    # Bank of Montreal
    "DUK",    # Duke Energy
    "DELL",   # Dell Technologies
    "BP",     # BP
    "VRT",    # Vertiv Holdings
    "GD",     # General Dynamics
    "BN",     # Brookfield Corporation
    "FCX",    # Freeport-McMoRan
    "WM",     # Waste Management
    "UPS",    # UPS
    "CRWD",   # CrowdStrike
    "EQIX",   # Equinix
    "ITUB",   # Itaú Unibanco
    "ICE",    # Intercontinental Exchange
    "WMB",    # Williams Companies
    "CM",     # Canadian Imperial Bank of Commerce
    "NGG",    # National Grid
    "BNS",    # Bank of Nova Scotia
    "CNQ",    # Canadian Natural Resources
    "WDC",    # Western Digital
    "FDX",    # FedEx
    "NKE",    # Nike
    "MELI",   # MercadoLibre
    "JCI",    # Johnson Controls
    "AMT",    # American Tower
    "SHW",    # Sherwin-Williams
    "MAR",    # Marriott International
    "ADP",    # Automatic Data Processing
    "PNC",    # PNC Financial
    "PWR",    # Quanta Services
    "ECL",    # Ecolab
    "EMR",    # Emerson Electric
    "MMM",    # 3M
    "STX",    # Seagate Technology
    "USB",    # US Bancorp
    "ITW",    # Illinois Tool Works
    "CDNS",   # Cadence Design Systems
    "MCO",    # Moody's
    "BK",     # Bank of New York Mellon
    "RCL",    # Royal Caribbean
    "SNPS",   # Synopsys
    "REGN",   # Regeneron Pharmaceuticals
    "KKR",    # KKR & Co
    "CTAS",   # Cintas
    "MSI",    # Motorola Solutions
    "CSX",    # CSX
    "CMI",    # Cummins
    "ABNB",   # Airbnb
    "ORLY",   # O'Reilly Automotive
    "MNST",   # Monster Beverage
    "CL",     # Colgate-Palmolive
]
