import twint

#Configure
c = twint.Config()
c.Search = "covid"

#Run
twint.run.Search(c)