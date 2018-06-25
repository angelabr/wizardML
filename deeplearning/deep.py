import numpy as np
import random
from copy import deepcopy

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
# Class for eClassifiers
# Attributes: Id, type of classifier, name, superType if it has one, features as in features it contains
class eClassifiers:
	
	def __init__(self, id, type, name, eSuperType, features):
		
		self.id = id
		self.type = type
		self.name = name
		self.eSuperType = eSuperType
		self.features = features

# Class for eStructures
# Attributes: Id, type of structure, name, eType, eClassifier class where they belong

class eStructures:
	
	def __init__(self, id, type, name, eType, eClassifier):
		self.id = id
		self.type = type
		self.name = name
		self.eType = eType
		self.eClassifier = eClassifier

# Class for errors found in the model
# Id, Location as in where is the origin of the error, affectsTo because it can affect the origin itself or something inside it 
# as features inside a class, idAffected references which concrete element is wrong, errorType as in which type of error is
class errorClass:

	def __init__(self, id, location, affectsTo, idAffected, solved, errorType):
		self.id = id
		self.location = location
		self.affectsTo = affectsTo
		self.idAffected = idAffected
		self.solved = solved
		self.errorType = errorType


# This method simulates the analysis of the model to find new errors
def error_checker(error, action):
	fnames = []
	solved = False
	if error.affectsTo is "features":
		# Empty type in attribute
		for e in error.location.features:
			fnames.append(e.name)
			if action == 3:
				if e.eType is "":
					solved = False
				else:
					if error.errorType is "type":
						solved = True
			
		# Repeated names in attributes
		if action == 2:		
			if len(fnames) != len(set(fnames)):
				solved = False
			else:
				if error.errorType is "name":
					solved = True
					print("NAMES NOT REPEATED")

	if error.affectsTo is "classifiers":
		# Recursive super type
		if action == 1:
			if error.location.eSuperType is error.location.name:
				solved = False
			else:
				solved = True
				print("SUPER TYPE FIXED")
	 				
	return solved

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
# Define transitions, what happens in the model when an action happens
# An action will modify errors (if successful) and the model

def act(error, action):

# Definition of what each action does
# Each action modifies model so it is needed to check errors again
	AUX_DEGREE = 0
	DEGREE = len(global_errors)
	id_ = 0
	location_ = 0
	affectsTo_ = 0
	idAffected_ = 0
	solved_ = False
	fnames = []
	#if error.affectsTo is "class":
	if action == 0:
		#This actually modify the class name
		print("ACTION 0 - RENAME CLASS")
		id_ = error.id
		location_ = eClassifiers(id=error.location.id, type=error.location.type, name = error.location.name + str(random.randint(1,9999)), 
			eSuperType = error.location.eSuperType, features = error.location.features)
		affectsTo_ = error.affectsTo
		idAffected_ = error.idAffected
		errorType_ = error.errorType

	if action == 1:
		# Delete recursive supertype
		print("ACTION 1 - DELETE STYPE")
		error.location.eSuperType = ""
		id_ = error.id
		location_ = eClassifiers(id=error.location.id, type=error.location.type, name = error.location.name, 
			eSuperType = "", features = error.location.features)
		affectsTo_ = error.affectsTo
		idAffected_ = eClassifiers(id=error.location.id, type=error.location.type, name = error.location.name, 
			eSuperType = "", features = error.location.features)
		errorType_ = error.errorType

	if action == 2:
		# Rename attribute with name duplicated
		print("ACTION 2 - RENAME ATTRIBUTE")
		id_ = error.id
		location_ = error.location
		affectsTo_ = error.affectsTo
		if error.affectsTo is "features":
			idAffected_ = eStructures(id = error.idAffected.id, type = error.idAffected.type, name = error.idAffected.name+ str(random.randint(1,9999)),
			 eType = error.idAffected.eType, eClassifier = error.idAffected.eClassifier)
			error.location.features[error.idAffected.id].name = error.idAffected.name + str(random.randint(1,9999))
		else:
			idAffected = idAffected_
		errorType_ = error.errorType

	if action == 3:
		print("ACTION 3 - MODIFY TYPE")
		# Mock-up of this... to change later
		# It assigns a type to an attribute without type
		id_ = error.id
		location_ = error.location
		affectsTo_ = error.affectsTo
		if error.affectsTo is "features":
			idAffected_ = eStructures(id = error.idAffected.id, type = error.idAffected.type, name = error.idAffected.name,
			 eType = "EMock", eClassifier = error.idAffected.eClassifier)
			error.location.features[error.idAffected.id].eType = idAffected_.eType
		else:
			idAffected = idAffected_	
		errorType_ = error.errorType

		
	
	aux_error = errorClass(id = id_, location = location_, affectsTo = affectsTo_, idAffected = idAffected_, solved = solved_, errorType = errorType_)
	#Checks if the action has solved the error
	aux_error.solved = error_checker(aux_error, action)
	#If it was solved it is removed from the list of errors
	if aux_error.solved == True:
		global_errors.remove(error)

	AUX_DEGREE = len(global_errors)
	print("Updated errors")
	print (AUX_DEGREE)
	print("General errors")
	print (DEGREE)

	#So if there are the same number of problems as before or more reward decreases
	if AUX_DEGREE >= DEGREE:
		reward = -25
		print("Reward: "+str(reward))
		done = False
		
	# Less number of errors but still some, some reward but not done yet	
	if (AUX_DEGREE < DEGREE) and (AUX_DEGREE != 0):
		reward = 500	
		DEGREE -=1
		print("Reward: "+str(reward))
		done = False

	# If no more errors, maximum reward and done
	if AUX_DEGREE == 0:
		reward = 10000
		DEGREE = 0
		print("Reward: "+str(reward))
		done = True

	return aux_error, reward, done

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

# Initilize the model to be analyzed
# In a final version this should get the model from EMF
# Parsing the XML into a Python datastructure
# In here we assume the XML has been parsed and we create our data structure
# Example 1 : One class with two attributes with the same name
# Example 2 : Ex1 + Class with attribute without type 
# We don't take into account relations/references for now
# Example 3 : EX2 + Class with self supertype
eC1 = eClassifiers(id = 1, type = "EClass", name = "Web", eSuperType = "", features = [])
eS1 = eStructures(id = 0, type = "EAttribute", name = "name", eType = "EString", eClassifier = eC1)
eS2 = eStructures(id = 1, type = "EAttribute", name = "name", eType = "EInt", eClassifier = eC1)

eC2 = eClassifiers(id = 2, type = "EClass", name = "Webpage", eSuperType="", features = [])
eS3 = eStructures(id = 0, type = "EAttribute", name = "created", eType = "", eClassifier = eC2)

eC3 = eClassifiers(id = 3, type = "EClass", name = "Section", eSuperType="Section", features = [])


eC1.features = [eS1, eS2]
eC2.features = [eS3]

# Save names within class
fnames = []

model = [eC1, eC2, eC3]

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

# Initialize state of the model, errors found
# Again, we assume that this is provide somehow by EMF and introduced in a datastructure
# Or we can build a error checker inspired by Ecore in Python

error1 = errorClass(id = 0, location = eC1, affectsTo = "features", idAffected = eS1, solved = False, errorType = "name")
error2 = errorClass(id = 1, location = eC2, affectsTo = "features", idAffected = eS3, solved = False, errorType = "type")
error3 = errorClass(id = 2, location = eC3, affectsTo = "classifiers", idAffected = eC3, solved = False, errorType = "stype")


global_errors = [error1, error2, error3]

CONSISTENT = True
AUX_DEGREE = 0

if len(global_errors) >= 1:
	CONSISTENT = False

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

# Initialize actions
# Again this can be extracted somehow from EMF or provided by a set created in Python
# Actions available are divided by the model's dimensions (classes, attributes...) in order to reduce state space
# For this example we assume the algorithm cannot add or delete elements (superType as exception) by itself

RENAME_CLASS = 0
DELETE_STYPE = 1

EC_ACTIONS = [RENAME_CLASS, DELETE_STYPE]

RENAME_ATTRIBUTE = 2
MODIFY_TYPE = 3

EF_ACTIONS = [RENAME_ATTRIBUTE, MODIFY_TYPE]	
ACTIONS = [RENAME_CLASS, DELETE_STYPE, RENAME_ATTRIBUTE, MODIFY_TYPE]

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#Q LEARNING Algorithm

random.seed(42) # for reproducibility

N_EPISODES = 20

MAX_EPISODE_STEPS = 2

MIN_ALPHA = 0.02

alphas = np.linspace(1.0, MIN_ALPHA, N_EPISODES)
gamma = 1.0
eps = 0.2

q_table = dict()

def q(error, action=None):
	
	if error not in q_table:
		q_table[error] = np.zeros(len(ACTIONS))
		
	if action is None:
		return q_table[error]
	
	return q_table[error][action]


def choose_action(error):

	if random.uniform(0, 1) < eps:
		return random.choice(ACTIONS) 
	else:
		return np.argmax(q(error))	


for e in range(N_EPISODES):
	print("NUMBER OF GLOBAL_ERRORS")
	print(len(global_errors))
	total_reward = 0
	alpha = alphas[e]

	for state in global_errors:
		print("ERROR ID")
		print(state.id)

		for _ in range(MAX_EPISODE_STEPS):
			action = choose_action(state)
			next_state, reward, done = act(state, action)
			total_reward += reward
			q(state)[action] = q(state, action) + \
					alpha * (reward + gamma *  np.max(q(next_state)) - q(state, action))
			state = next_state
			if done:
				break		
				
	print(f"Episode {e + 1}: total reward -> {total_reward}")

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------