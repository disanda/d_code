from nltk.corpus import wordnet as wn

print('-----------')# 同义词集（synsets）
print(wn.synsets("car"))


print('-----------')# 查看第一个同义词集的定义与例句
s = wn.synsets("car")[0]
print(s.definition())
print(s.examples())

print('-----------')# 查看词语的上下义词（hypernyms）和下义词（hyponyms）
print(s.hypernyms())
print(s.hyponyms())


# ----------- 输出：
# [Synset('car.n.01'), Synset('car.n.02'), Synset('car.n.03'), Synset('car.n.04'), Synset('cable_car.n.01')]
# -----------
# a motor vehicle with four wheels; usually propelled by an internal combustion engine
# ['he needs a car to get to work']
# -----------
# [Synset('motor_vehicle.n.01')]
# [Synset('ambulance.n.01'), Synset('beach_wagon.n.01'), Synset('bus.n.04'), 
# Synset('cab.n.03'), Synset('compact.n.03'), Synset('convertible.n.01'), 
# Synset('coupe.n.01'), Synset('cruiser.n.01'), Synset('electric.n.01'), 
# Synset('gas_guzzler.n.01'), Synset('hardtop.n.01'), Synset('hatchback.n.01'), 
# Synset('horseless_carriage.n.01'), Synset('hot_rod.n.01'), Synset('jeep.n.01'), 
# Synset('limousine.n.01'), Synset('loaner.n.02'), Synset('minicar.n.01'), S
# ynset('minivan.n.01'), Synset('model_t.n.01'), Synset('pace_car.n.01'), 
# Synset('racer.n.02'), Synset('roadster.n.01'), Synset('sedan.n.01'), 
# Synset('sport_utility.n.01'), Synset('sports_car.n.01'), Synset('stanley_steamer.n.01'), 
# Synset('stock_car.n.01'), Synset('subcompact.n.01'), Synset('touring_car.n.01'),
# Synset('used-car.n.01')]
