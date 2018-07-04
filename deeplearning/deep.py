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
	cnames = []
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
			#It's not worth to fix this in a better way right now..
			if error.errorType is "name" and len(fnames) == 4 and len(set(fnames)) == 3:
				solved = True
			else:	
				if len(fnames) != len(set(fnames)):
					solved = False
				else:
					if error.errorType is "name":
						solved = True

	if error.affectsTo is "classifiers":
		# Recursive super type
		if action == 1:

			if error.location.eSuperType is error.location.name:
				solved = False
			else:
				if (error.errorType is "stype"):
					solved = True
					print("SOLVED")

		if action == 0:
			if isinstance(error.idAffected, list):
				print("SOY LISTA")
				if len(error.idAffected) > 1:
					print("VARIOS")
					for c in error.idAffected:
						cnames.append(c.name)
					print(len(cnames))
					print(len(set(cnames)))	
					if len(cnames) != len(set(cnames)):
						solved = False
					else:
						print("ELSE")
						if error.errorType is "namec":
							solved = True
							print("SOLVED")

	 				
	return solved

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
# Define transitions, what happens in the model when an action happens
# An action will modify errors (if successful) and the model

def act(error, action, list_errors):

# Definition of what each action does
# Each action modifies model so it is needed to check errors again
	AUX_DEGREE = 0
	DEGREE = len(list_errors)
	id_ = 0
	location_ = 0
	affectsTo_ = 0
	idAffected_ = 0
	solved_ = False
	fnames = []
	error_ = 0
	done = False
	#if error.affectsTo is "class":
	if action == 0:
		#This actually modify the class name
		print("ACTION 0 - RENAME CLASS")
		id_ = error.id
		if error.errorType is "namec":
			print("NAMEC")
			location_ = eClassifiers(id=error.location.id, type=error.location.type, name = error.location.name + str(random.randint(1,9999)), 
				eSuperType = error.location.eSuperType, features = error.location.features)
			if error.location.eSuperType is error.location.name:
				location_.eSuperType = location_.name
			if isinstance(error.idAffected, list):
				if len(error.idAffected) > 1:
					for g in list_errors:
						if g.id == error.id:
							error.idAffected.remove(g.location)
					error.idAffected.insert(0, location_)
					idAffected_ = error.idAffected
			else:
				idAffected_ = location_
		else:
			location_ = error.location
			idAffected_ = error.idAffected	
		affectsTo_ = error.affectsTo
		errorType_ = error.errorType

	if action == 1:
		# Delete recursive supertype
		print("ACTION 1 - DELETE STYPE")
		error.location.eSuperType = ""
		id_ = error.id
		location_ = eClassifiers(id=error.location.id, type=error.location.type, name = error.location.name, 
			eSuperType = "", features = error.location.features)
		affectsTo_ = error.affectsTo
		if error.errorType is "stype":
			idAffected_ = eClassifiers(id=error.location.id, type=error.location.type, name = error.location.name, 
			eSuperType = "", features = error.location.features)
		else:
			idAffected_ = error.idAffected
		errorType_ = error.errorType

	if action == 2:
		# Rename attribute with name duplicated
		print("ACTION 2 - RENAME ATTRIBUTE")
		id_ = error.id
		location_ = error.location
		affectsTo_ = error.affectsTo
		if error.errorType is "name":
			idAffected_ = eStructures(id = error.idAffected.id, type = error.idAffected.type, name = error.idAffected.name+ str(random.randint(1,9999)),
			 eType = error.idAffected.eType, eClassifier = error.idAffected.eClassifier)
			error.location.features[error.idAffected.id].name = error.idAffected.name + str(random.randint(1,9999))
		else:
			idAffected_ = error.idAffected
		errorType_ = error.errorType

	if action == 3:
		print("ACTION 3 - MODIFY TYPE")
		# Mock-up of this... to change later
		# It assigns a type to an attribute without type
		id_ = error.id
		location_ = error.location
		affectsTo_ = error.affectsTo
		if (error.affectsTo is "features"):
			idAffected_ = eStructures(id = error.idAffected.id, type = error.idAffected.type, name = error.idAffected.name,
			 eType = "EMock", eClassifier = error.idAffected.eClassifier)
			error.location.features[error.idAffected.id].eType = idAffected_.eType
		else:
			idAffected_ = error.idAffected
		errorType_ = error.errorType

		
	
	aux_error = errorClass(id = id_, location = location_, affectsTo = affectsTo_, idAffected = idAffected_, solved = solved_, errorType = errorType_)
	#Checks if the action has solved the error
	aux_error.solved = error_checker(aux_error, action)
	#If it was solved it is removed from the list of errors
	for g in list_errors:
		if g.id == aux_error.id and done!=True:
			if aux_error.solved == True and done!=True:
				list_errors.remove(g)
				done = True

	AUX_DEGREE = len(list_errors)
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
		done = True

	# If no more errors, maximum reward and done
	if AUX_DEGREE == 0:
		reward = 10000
		DEGREE = 0
		print("Reward: "+str(reward))
		done = True

	return aux_error, reward, done, list_errors

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

#------------------------- EXTRA MODEL 1 ------------------------------

eC1x = eClassifiers(id = 1, type = "EClass", name = "Car", eSuperType = "", features = [])
eS1x = eStructures(id = 0, type = "EAttribute", name = "name", eType = "EString", eClassifier = eC1)

eC2x = eClassifiers(id = 2, type = "EClass", name = "Car", eSuperType="", features = [])
eS2x = eStructures(id = 1, type = "EAttribute", name = "brand", eType = "EInt", eClassifier = eC1)
eS3x = eStructures(id = 0, type = "EAttribute", name = "brand", eType = "EString", eClassifier = eC2)

eC1x.features = [eS1x]
eC2x.features = [eS2x, eS3x]

model = [eC1x, eC2x]


error1x = errorClass(id = 3, location = eC1x, affectsTo = "classifiers", idAffected = [eC1x, eC2x], solved = False, errorType = "namec")
error2x = errorClass(id = 1, location = eC1x, affectsTo = "features", idAffected = eS1x, solved = False, errorType = "type")
error3x = errorClass(id = 0, location = eC2x, affectsTo = "features", idAffected = eS2x, solved = False, errorType = "name")


global_errorsx = [error1x, error2x, error3x]


#--------------------------------------------------------------------------------------------

#------------------------- EXTRA MODEL 2 ------------------------------

eC1w = eClassifiers(id = 1, type = "EClass", name = "Mega", eSuperType = "Mega", features = [])
eS1w = eStructures(id = 0, type = "EAttribute", name = "attr1", eType = "EString", eClassifier = eC1w)
eS2w = eStructures(id = 1, type = "EAttribute", name = "attr1", eType = "EInt", eClassifier = eC1w)
eS3w = eStructures(id = 2, type = "EAttribute", name = "attr2", eType = "EString", eClassifier = eC1w)
eS4w = eStructures(id = 3, type = "EAttribute", name = "attr2", eType = "EInt", eClassifier = eC1w)

eC1w.features = [eS1w, eS2w, eS3w, eS4w]



error1w = errorClass(id = 2, location = eC1w, affectsTo = "classifiers", idAffected = eC1w, solved = False, errorType = "stype")
# below would need a list for idAffected
error2w = errorClass(id = 0, location = eC1w, affectsTo = "features", idAffected = eS1w, solved = False, errorType = "name")
error3w = errorClass(id = 0, location = eC1w, affectsTo = "features", idAffected = eS3w, solved = False, errorType = "name")


global_errorsw = [error1w, error2w, error3w]


#--------------------------------------------------------------------------------------------
#------------------------- EXTRA MODEL 3 ------------------------------

eC1y = eClassifiers(id = 1, type = "EClass", name = "Mega2", eSuperType = "", features = [])
eS1y = eStructures(id = 0, type = "EAttribute", name = "attr1", eType = "EString", eClassifier = eC1y)
eS2y = eStructures(id = 1, type = "EAttribute", name = "attr1", eType = "", eClassifier = eC1y)

eC1y.features = [eS1y, eS2y]



error1y = errorClass(id = 0, location = eC1y, affectsTo = "features", idAffected = eS1y, solved = False, errorType = "name")
error2y = errorClass(id = 1, location = eC1y, affectsTo = "features", idAffected = eS2y, solved = False, errorType = "type")


global_errorsy = [error1y, error2y]


#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#------------------------- EXTRA MODEL 4 ------------------------------

eC1z = eClassifiers(id = 1, type = "EClass", name = "Mega3", eSuperType = "Mega3", features = [])
eC2z = eClassifiers(id = 1, type = "EClass", name = "Mega3", eSuperType = "Mega3", features = [])

eS1z = eStructures(id = 0, type = "EAttribute", name = "attr1", eType = "EInt", eClassifier = eC1z)
eS2z = eStructures(id = 1, type = "EAttribute", name = "attr1", eType = "", eClassifier = eC1z)
eS3z = eStructures(id = 0, type = "EAttribute", name = "attr2", eType = "EInt", eClassifier = eC2z)
eS4z = eStructures(id = 1, type = "EAttribute", name = "attr2", eType = "EString", eClassifier = eC2z)

eC1z.features = [eS1z, eS2z]
eC2z.features = [eS3z, eS4z]


error1z = errorClass(id = 3, location = eC1z, affectsTo = "classifiers", idAffected = [eC1z, eC2z], solved = False, errorType = "namec")
error2z = errorClass(id = 0, location = eC1z, affectsTo = "features", idAffected = eS1z, solved = False, errorType = "name")
error3z = errorClass(id = 1, location = eC1z, affectsTo = "features", idAffected = eS2z, solved = False, errorType = "type")
error4z = errorClass(id = 2, location = eC1z, affectsTo = "classifiers", idAffected = eC1z, solved = False, errorType = "stype")
error5z = errorClass(id = 0, location = eC2z, affectsTo = "features", idAffected = eS3z, solved = False, errorType = "name")
error6z = errorClass(id = 2, location = eC2z, affectsTo = "classifiers", idAffected = eC2z, solved = False, errorType = "stype")

global_errorsz = [error1z, error2z, error3z, error4z, error5z, error6z]


#--------------------------------------------------------------------------------------------

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

N_EPISODES = 10

MAX_EPISODE_STEPS = 3

MIN_ALPHA = 0.02

alphas = np.linspace(1.0, MIN_ALPHA, N_EPISODES)
gamma = 1.0
eps = 0.1

q_table = dict()

queue = [global_errors, global_errorsx, global_errorsw, global_errorsy, global_errorsz]

def q(error, action=None):
	
	if error.id not in q_table:
		q_table[error.id] = np.zeros(len(ACTIONS))
		
	if action is None:
		return q_table[error.id]
	
	return q_table[error.id][action]


def choose_action(error):

	if random.uniform(0, 1) < eps:
		print("RANDOM")
		return random.choice(ACTIONS) 
	else:
		return np.argmax(q(error))	

for errors in queue:
	print("QUEUE LEN")
	print(len(errors))
	for e in range(N_EPISODES):
		print("NUMBER OF GLOBAL_ERRORS")
		print(len(errors))
		total_reward = 0
		alpha = alphas[e]

		for state in errors:
			print("ERROR ID")
			print(state.id)

			for w in range(MAX_EPISODE_STEPS):
				print("STEP")
				print(w)
				action = choose_action(state)
				r = q(state)
				print(f"rename_class={r[RENAME_CLASS]}, delete_stype={r[DELETE_STYPE]}, rename_attrb={r[RENAME_ATTRIBUTE]}, modify_type={r[MODIFY_TYPE]}")
				next_state, reward, done, errors = act(state, action, errors)
				total_reward += reward
				if done:
					w = MAX_EPISODE_STEPS
				else:	
					q(state)[action] = q(state, action) + \
							alpha * (reward + gamma *  np.max(q(next_state)) - q(state, action))
					state = next_state
				if done:
					break		
					
		print(f"Episode {e + 1}: total reward -> {total_reward}")
	print("QUEUE COMPLETED")	
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------