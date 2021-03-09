"""

TO DO ELENA: figure out where to put this

"""

RAW_CSV_URL = {
    2017: {
        "L1C": {
            "frh01": "https://syncandshare.lrz.de/dl/fiA33ywfHQdzbxXwYQ5zLVpp/frh01.zip",
            "frh02": "https://syncandshare.lrz.de/dl/fi2pg7sXMjTQRSzrWxRdGLux/frh02.zip",
            "frh03": "https://syncandshare.lrz.de/dl/fiFbj4sqWYzd4kmcEThTJzYC/frh03.zip",
            "frh04": "https://syncandshare.lrz.de/dl/fi6rE9rgVyPqS5T5yN6Fccv5/frh04.zip",
        },
        "L2A": {
            "frh01": "https://syncandshare.lrz.de/dl/fiXr3Kho1KsK7gQ9a3DuyehH/frh01.zip",
            "frh02": "https://syncandshare.lrz.de/dl/fiSz3L41G8CkTJ2si9grVtcB/frh02.zip",
            "frh03": "https://syncandshare.lrz.de/dl/fiMygybgQtqUJ7x4EQexy4ZU/frh03.zip",
            "frh04": "https://syncandshare.lrz.de/dl/fiSB111Gsx4v1c4HT4Gh4jY/frh04.zip",
            
        }
    },
    2018: {
        "L1C": {
            "frh01": "https://syncandshare.lrz.de/dl/fiUTbmrK3yXyshiQBXkvr9ZX/frh01.zip",
            "frh02": "https://syncandshare.lrz.de/dl/fiFJZn7uBmGkdJAKn47edxqg/frh02.zip",
            "frh03": "https://syncandshare.lrz.de/dl/fi74422CywCLeoSYa7QNNA9X/frh03.zip",
            "frh04": "https://syncandshare.lrz.de/dl/fiDvCLTvC93Xky6UdVbe3Ps6/frh04.zip",
        }
    }
}

INDEX_FILE_URLs = {
    2017: {
        "L1C": {
            "frh01": "https://syncandshare.lrz.de/dl/fiE7ExSPEF5j1LHADGZ1GcAV/frh01.csv",
            "frh02": "https://syncandshare.lrz.de/dl/fiEutoBWs3JFjCfJpoLWq5HF/frh02.csv",
            "frh03": "https://syncandshare.lrz.de/dl/fiJL3LMrzYwmULbvzFiyVZuY/frh03.csv",
            "frh04": "https://syncandshare.lrz.de/dl/fiCntufUMakKdjWZNq8eS5vw/frh04.csv",
            "belle-ile": "https://syncandshare.lrz.de/dl/fiM9X4VgWM8qb7F5Z4TVHa8A/belle-ile.csv"
        },
        "L2A": {
            "frh01": "https://syncandshare.lrz.de/dl/fiPZ7vpN4zVQAR9gqmqv12Ej/frh01.csv",
            "frh02": "https://syncandshare.lrz.de/dl/fi1749MFuYgWc4qKwrVWu2hR/frh02.csv",
            "frh03": "https://syncandshare.lrz.de/dl/fiTKuWmkwnxRoP84V3gWMpqs/frh03.csv",
            "frh04": "https://syncandshare.lrz.de/dl/fiDmi5qobgbfJHJLNEPf99xT/frh04.csv",
            "belle-ile": "https://syncandshare.lrz.de/dl/fiVn3VpGhLETZkaRnEBDgUad/belle-ile.csv"
        }
    },
    2018: {
            "L1C": {
                "frh01": "https://syncandshare.lrz.de/dl/fiWdBJxymGfNMJ4AfRZLS1vf/frh01.csv",
                "frh02": "https://syncandshare.lrz.de/dl/fiCMvKxLh5KHSjHXoesgVvs6/frh02.csv",
                "frh03": "https://syncandshare.lrz.de/dl/fiXNTFuoox7agRzYYZv9uurh/frh03.csv",
                "frh04": "https://syncandshare.lrz.de/dl/fiRMLCLwLamgzWKwtTeWienw/frh04.csv",
            }
        }
}

SHP_URLs = {
    2017: {
        "frh01": "https://syncandshare.lrz.de/dl/fiAHe7ZYMerBi2yJ5hKJmTXS/frh01.tar.gz",
        "frh02": "https://syncandshare.lrz.de/dl/fi8L5iwpJXs2b9hKEFjQoML5/frh02.tar.gz",
        "frh03": "https://syncandshare.lrz.de/dl/fiTdWAa8b9K4XVmrBbZ6413i/frh03.tar.gz",
        "frh04": "https://syncandshare.lrz.de/dl/fiKfoL1VW9jiDXPgnVXu7ZFK/frh04.tar.gz",
        "belle-ile": "https://syncandshare.lrz.de/dl/fiVKniyfTVARvAhZfVxq7keW/belle-ile.tar.gz"
    },
    2018: {
        "frh01": "https://syncandshare.lrz.de/dl/fiUe1VMZrrbcsK82TKJ9EFi2/frh01.tar.gz",
        "frh02": "https://syncandshare.lrz.de/dl/fi7CAQab5f8MGp3dc4NdnpWg/frh02.tar.gz",
        "frh03": "https://syncandshare.lrz.de/dl/fiQBZMk2vCPVJuuRifHiTqeX/frh03.tar.gz",
        "frh04": "https://syncandshare.lrz.de/dl/fiUpxJimscvcTi6aBx7r4sEz/frh04.tar.gz",
    }
}

FILESIZES = {
    2017: {
        "L1C": {
            "frh01": 2559635960,
            "frh02": 2253658856,
            "frh03": 2493572704,
            "frh04": 1555075632,
            "belle-ile": 17038944
        },
        "L2A": {
            "frh01": 987253072,
            "frh02": 803455680,
            "frh03": 890016904,
            "frh04": 639211696,
            "belle-ile": 8037952
        }
    },
    2018: {
        "L1C": {
            "frh01": 9878839310,
            "frh02": 8567550069,
            "frh03": 10196638286,
            "frh04": 6270634169
        }
    }
}

H5_URLs = {
    2017: {
        "L1C": {
            "frh01": "https://syncandshare.lrz.de/dl/fiFe2C3qDW5MWnVtWdaAT7xC/frh01.h5.tar.gz",
            "frh02": "https://syncandshare.lrz.de/dl/fi3dyXpipntJyiCZZJdLNcTi/frh02.h5.tar.gz",
            "frh03": "https://syncandshare.lrz.de/dl/fi8ahoBEbekCKh61PxDAvjQ/frh03.h5.tar.gz",
            "frh04": "https://syncandshare.lrz.de/dl/fi77rzsEJMWXumq3jpi1VPYF/frh04.h5.tar.gz",
            "belle-ile": "https://syncandshare.lrz.de/dl/fiTMJz5aByhqtMZShJoBmSnY/belle-ile.h5.tar.gz"
        },
        "L2A": {
            "frh01": "https://syncandshare.lrz.de/dl/fiLH5KGLKXedTSTTjosdHvar/frh01.h5",
            "frh02": "https://syncandshare.lrz.de/dl/fiYbtMkrVFGM2UDGez24QJt6/frh02.h5",
            "frh03": "https://syncandshare.lrz.de/dl/fiT5efJwbktMTMbRsr8cvsuh/frh03.h5",
            "frh04": "https://syncandshare.lrz.de/dl/fiTuAYAMGPzYd9VTKtSmsFzE/frh04.h5",
            "belle-ile": "https://syncandshare.lrz.de/dl/fiMBX3rfpVjVHPSQH4fhtyna/belle-ile.h5.tar.gz"
        }
    }
}

# 9-classes used in ISPRS submission
CLASSMAPPINGURL = "https://syncandshare.lrz.de/dl/fiWcv23b3PxswYZFh2htEpSs/classmapping.csv"

# 13-classes used in ICML workshop
CLASSMAPPINGURL_ICML = "https://syncandshare.lrz.de/dl/fiAXzNVSgAz7sKBdonhsCpkG/classmapping_icml.csv"

CODESURL = "https://syncandshare.lrz.de/dl/fiFVnHYsEsix7HTGYRh6Zh3/codes.csv"
