#include <vector>
#include <string>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "CoarseBiLM.h"
#include "moses/ScoreComponentCollection.h"
#include "moses/Hypothesis.h"
#include "moses/Manager.h"
#include "moses/InputType.h"
#include "moses/Word.h"
#include "moses/AlignmentInfo.h"
#include "lm/model.hh"

using namespace std;
using namespace lm::ngram;

namespace Moses {
int CoarseBiLMState::Compare(const FFState& other) const {
	const CoarseBiLMState &otherState =
			static_cast<const CoarseBiLMState&>(other);

	if (m_targetLen == otherState.m_targetLen)
		return 0;
	return (m_targetLen < otherState.m_targetLen) ? -1 : +1;
}
////////////////////////////////////////////////////////////////
CoarseBiLM::CoarseBiLM(const std::string &line) :
		StatefulFeatureFunction(3, line) {
	//FactorCollection& factorFactory = FactorCollection::Instance(); //Factor Factory to use for BOS_ and EOS_
}

CoarseBiLM::~CoarseBiLM() {
	delete CoarseLM;
  delete srcWordToClusterId;
  delete tgtWordToClusterId;
  delete bitokenToBitokenId;
  delete bitokenIdToClusterId;
}

void CoarseBiLM::Load() {
	cvtSrcToClusterId = false;
	cvtBitokenToBitokenId = false;
	cvtBitokenIdToClusterId = false;
	ReadParameters();
	readLanguageModel(m_lmPath.c_str());
}

void CoarseBiLM::EvaluateInIsolation(const Phrase &source,
		const TargetPhrase &targetPhrase,
		ScoreComponentCollection &scoreBreakdown,
		ScoreComponentCollection &estimatedFutureScore) const {
}

void CoarseBiLM::EvaluateWithSourceContext(const InputType &input,
		const InputPath &inputPath, const TargetPhrase &targetPhrase,
		const StackVec *stackVec, ScoreComponentCollection &scoreBreakdown,
		ScoreComponentCollection *estimatedFutureScore) const {
}

void CoarseBiLM::EvaluateTranslationOptionListWithSourceContext(
		const InputType &input,
		const TranslationOptionList &translationOptionList) const {
}

//load the sri language models
/*
 * I need to load the following many to one maps:
 * 1. english (tgt) 400 cluster ids. Map<Word, ClusterId>
 * 2. chinese (src) 400 cluster ids. Map<Word, ClusterId>
 * 3. bitoken mapping Map<TgtId-SrcId, BitokenTag>
 * 4. bitoken clusters Map<BitokenTag, ClusterId>
 *
 * I also need to load the SRILM language model. Before calling the language model to get the score I need to
 * get the current target hypothesis and the source phrase that we are currently looking at. I also need the alignments
 * I then create the bitokens using those alignments and use the many to one maps to get to bitoken cluster id phrase.
 * I then call the language model on that phrase and return the score.
 *
 */

FFState* CoarseBiLM::EvaluateWhenApplied(const Hypothesis& cur_hypo,
		const FFState* prev_state,
		ScoreComponentCollection* accumulator) const {
	// dense scores

	vector<string> targetWords;
	vector<string> sourceWords;
	vector<string> targetWordIDs;
	vector<string> sourceWordIDs;
	vector<string> bitokens;
	vector<string> bitokenBitokenIDs;
	vector<string> bitokenWordIDs;

	float totalScore = 0.0;
	std::map<int, std::vector<int> > alignments;

	TargetPhrase currTargetPhrase = cur_hypo.GetCurrTargetPhrase();
	Manager& manager = cur_hypo.GetManager();
	const Sentence& sourceSentence =
			static_cast<const Sentence&>(manager.GetSource());

	//Get target words. Also, get the previous hypothesised target words.
	getTargetWords(cur_hypo, targetWords, alignments);
	replaceWordsWithClusterID(targetWords, tgtWordToClusterId, targetWordIDs);
	vector<string> wordsToScore = targetWordIDs;

	if (cvtSrcToClusterId) {
		//Reads the source sentence and fills the sourceWords vector wit source words.
		getSourceWords(sourceSentence, sourceWords);
		replaceWordsWithClusterID(sourceWords, srcWordToClusterId,
				sourceWordIDs);
		if (cvtBitokenToBitokenId) {
			//Create bitokens.
			createBitokens(sourceWordIDs, targetWordIDs, alignments, bitokens);
			//Replace bitokens with bitoken tags
			replaceWordsWithClusterID(bitokens, bitokenToBitokenId,
					bitokenBitokenIDs);
			wordsToScore = bitokenBitokenIDs;
			if (cvtBitokenIdToClusterId) {
				//Replace bitoken tags with bitoken cluster ids
				replaceWordsWithClusterID(bitokenBitokenIDs,
						bitokenIdToClusterId, bitokenWordIDs);
				wordsToScore = bitokenWordIDs;
			}
		}
	}

	State state(CoarseLM->BeginSentenceState()), outState;

	//std::cerr << "Scoring Words" << std::endl;
	for (std::vector<std::string>::const_iterator iterator =
			wordsToScore.begin(); iterator != wordsToScore.end(); iterator++) {
		std::string word = *iterator;
		float score = CoarseLM->Score(state, word, outState);
		//std::cerr << "Word: " << word << ", score: " << score << std::endl;
		totalScore = totalScore + score;
		state = outState;
	}

	/*std::cerr << "### Printing Alignments ###" << std::endl;
	 for(std::map<int, std::vector<int> >::const_iterator it = alignments.begin(); it != alignments.end(); it++) {
	 std::cerr << it->first << ":";
	 for(vector<int>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); it2++) {
	 std::cerr << " " << *it2;
	 }
	 std::cerr << std::endl;
	 }

	 std::cerr << "### Printing Target Words ###" << std::endl;
	 printList(targetWords);

	 std::cerr << "### Printing Source Words ###" << std::endl;
	 printList(sourceWords);

	 std::cerr << "### Printing Target Words Cluster IDs###" << std::endl;
	 printList(targetWordIDs);

	 std::cerr << "### Printing Source Words Cluster IDs###" << std::endl;
	 printList(sourceWordIDs);

	 std::cerr << "### Printing Bitokens ###" << std::endl;
	 printList(bitokens);

	 std::cerr << "### Printing Bitoken-Bitoken IDs ###" << std::endl;
	 printList(bitokenBitokenIDs);

	 if (cvtBitokenIdToClusterId) {
	 std::cerr << "#### FEATURE FUNCTION: CoarseBiLM" << std::endl;
	 } else if (cvtBitokenToBitokenId) {
	 std::cerr << "#### FEATURE FUNCTION: CoarseBiLMWithoutClustering" << std::endl;
	 } else {
	 std::cerr << "#### FEATURE FUNCTION: CoarseLM" << std::endl;
	 }
	 std::cerr << "### Printing wordsToScore Cluster IDs ###" << std::endl;
	 printList(wordsToScore);

	 std::cerr << "### Probabilties ###" << std::endl;

	 std::cerr << "Total Score: " << totalScore << std::endl;
	 std::cerr << "numScoreComponents: " << m_numScoreComponents << std::endl;
	 std::cerr << "### Done For This Sentence ###" << std::endl;
	 */

	vector<float> newScores(m_numScoreComponents);
	newScores[0] = totalScore;
	accumulator->PlusEquals(this, newScores);

	size_t newState = getState(cur_hypo);
	return new CoarseBiLMState(newState);
}

/*
 * Get the target words from the current hypothesis and also the previous words from previous hypothesis.
 * While doing this, also get the alignments for the target words to the source words.
 *
 * targetWords and alignments are populated in this method.
 */
void CoarseBiLM::getTargetWords(const Hypothesis& cur_hypo,
		std::vector<std::string> &targetWords,
		std::map<int, vector<int> > &alignments) const {

	int currentTargetPhraseSize = cur_hypo.GetCurrTargetPhrase().GetSize();
	int previousWordsNeeded = nGramOrder - currentTargetPhraseSize;

	if (previousWordsNeeded > 0) {
		vector<string> previousWords(previousWordsNeeded);
		//Get previous target words
		getPreviousTargetWords(cur_hypo, previousWordsNeeded, previousWords,
				alignments);

		for (int i = previousWords.size() - 1; i >= 0; i--) {
			string previousWord = previousWords[i];
			boost::algorithm::trim(previousWord);
			if (!previousWord.empty()) {
				targetWords.push_back(previousWords[i]);
			}
		}
	}

	std::size_t targetBegin = cur_hypo.GetCurrTargetWordsRange().GetStartPos();
	for (int index = 0; index < currentTargetPhraseSize; index++) {
		string word = cur_hypo.GetWord(targetBegin + index).ToString();
		boost::algorithm::trim(word);
		targetWords.push_back(word);
		//find alignments for current word
		std::set<size_t> currWordAlignments =
				cur_hypo.GetCurrTargetPhrase().GetAlignTerm().GetAlignmentsForTarget(
						index + targetBegin);
		size_t sourceBegin = cur_hypo.GetCurrSourceWordsRange().GetStartPos();

		//add alignments to map
		for (std::set<size_t>::const_iterator iterator =
				currWordAlignments.begin();
				iterator != currWordAlignments.end(); iterator++) {
			std::map<int, vector<int> >::iterator it = alignments.find(
					index + targetBegin);
			vector<int> alignedSourceIndices;
			if (it != alignments.end()) {
				//found vector of indices
				alignedSourceIndices = it->second;
			}
			alignedSourceIndices.push_back((*iterator) + sourceBegin);
			alignments[index + targetBegin] = alignedSourceIndices;
		}
	}

}

/*
 * Get previous n (previousWordsNeeded) words from previous hypothesis. Also  get the alignments for those words.
 * targetWords and alignments are populated.
 */
void CoarseBiLM::getPreviousTargetWords(const Hypothesis& cur_hypo,
		int previousWordsNeeded, std::vector<std::string> &targetWords,
		std::map<int, vector<int> > &alignments) const {
	const Hypothesis * prevHypo = cur_hypo.GetPrevHypo();
	int found = 0;

	while (prevHypo && found != previousWordsNeeded) {
		const TargetPhrase& currTargetPhrase = prevHypo->GetCurrTargetPhrase();
		size_t tpBegin = prevHypo->GetCurrTargetWordsRange().GetStartPos();
		size_t sourceBegin = prevHypo->GetCurrSourceWordsRange().GetStartPos();

		for (int i = currTargetPhrase.GetSize() - 1; i > -1; i--) {
			if (found != previousWordsNeeded) {
				const Word& word = currTargetPhrase.GetWord(i);
				targetWords[found] = word.ToString();

				//find alignments for current word
				std::set<size_t> currWordAlignments =
						currTargetPhrase.GetAlignTerm().GetAlignmentsForTarget(
								i + tpBegin);
				//add alignments to map
				for (std::set<size_t>::const_iterator iterator =
						currWordAlignments.begin();
						iterator != currWordAlignments.end(); iterator++) {
					std::map<int, vector<int> >::iterator it = alignments.find(
							i + tpBegin);
					vector<int> alignedSourceIndices;
					if (it != alignments.end()) {
						alignedSourceIndices = it->second;
					}
					alignedSourceIndices.push_back((*iterator) + sourceBegin);
					alignments[i + tpBegin] = alignedSourceIndices;
				}

				found++;
			} else {
				return;
			}
		}
		prevHypo = prevHypo->GetPrevHypo();
	}
}

/*
 * Get the words in sourceSentence and fill the sourceWords vector.
 */
void CoarseBiLM::getSourceWords(const Sentence &sourceSentence,
		std::vector<std::string> &sourceWords) const {
	for (int index = 0; index < sourceSentence.GetSize(); index++) {
		string word = sourceSentence.GetWord(index).ToString();
		boost::algorithm::trim(word);
		sourceWords.push_back(word);
	}
}

void CoarseBiLM::replaceWordsWithClusterID(
		const std::vector<std::string> &words,
		const std::map<std::string, std::string> &clusterIdMap,
		std::vector<std::string> &wordClusterIDs) const {
//std::cerr << clusterIdMap.size() << std::endl;
	for (std::vector<std::string>::const_iterator it = words.begin();
			it != words.end(); it++) {
		std::string word = *it;
		boost::algorithm::trim(word);
		std::map<std::string, std::string>::const_iterator pos =
				clusterIdMap.find(word);
		if (pos == clusterIdMap.end()) {
			//std::cerr << "did not find a value: " << word << std::endl;
		} else {
			std::string clusterId = pos->second;
			wordClusterIDs.push_back(clusterId);
		}
	}
}

void CoarseBiLM::createBitokens(const std::vector<std::string> &sourceWords,
		const std::vector<std::string> &targetWords,
		const std::map<int, std::vector<int> > &alignments,
		std::vector<std::string> &bitokens) const {
	for (int index = 0; index < targetWords.size(); index++) {
		string targetWord = targetWords[index];
		string sourceWord = "";
		std::map<int, vector<int> >::const_iterator pos = alignments.find(
				index);
		if (pos == alignments.end()) {
			//std::cerr << "did not find a value: " << targetWord << std::endl;
			sourceWord = "NULL";
		} else {
			vector<int> sourceIndicess = pos->second;
			for (vector<int>::const_iterator it = sourceIndicess.begin();
					it != sourceIndicess.end(); it++) {
				sourceWord = sourceWord + "_" + sourceWords[*it];
			}
			sourceWord.erase(0, 1);
		}
		string bitoken = sourceWord + "-" + targetWord;
		//std::cerr << "Bitoken: " << bitoken << std::endl;
		bitokens.push_back(bitoken);
	}
}

size_t CoarseBiLM::getState(const Hypothesis& cur_hypo) const {

	int currentTargetPhraseSize = cur_hypo.GetCurrTargetPhrase().GetSize();
	int previousWordsNeeded = nGramOrder - currentTargetPhraseSize;
	size_t hashCode = 0;

	if (previousWordsNeeded > 0) {
		vector<string> previousWords(previousWordsNeeded);
		std::map<int, vector<int> > alignments;
		//Get previous target words
		getPreviousTargetWords(cur_hypo, previousWordsNeeded, previousWords,
				alignments);

		for (int i = previousWords.size() - 1; i >= 0; i--) {
			string previousWord = previousWords[i];
			boost::algorithm::trim(previousWord);
			if (!previousWord.empty()) {
				boost::hash_combine(hashCode, previousWords[i]);
			}
		}
	}

	std::size_t targetBegin = cur_hypo.GetCurrTargetWordsRange().GetStartPos();
	for (int index = 0; index < currentTargetPhraseSize; index++) {
		boost::hash_combine(hashCode,
				cur_hypo.GetWord(targetBegin + index).ToString());
	}

	return hashCode;
}

FFState* CoarseBiLM::EvaluateWhenApplied(const ChartHypothesis& /* cur_hypo */,
		int /* featureID - used to index the state in the previous hypotheses */,
		ScoreComponentCollection* accumulator) const {
	return new CoarseBiLMState(0);
}

void CoarseBiLM::SetParameter(const std::string& key,
		const std::string& value) {
	std::cerr << "Key: " + key << std::endl;
	if (key == "tgtWordToClusterId") {
		LoadManyToOneMap(value, tgtWordToClusterId);
	} else if (key == "srcWordToClusterId") {
		cvtSrcToClusterId = true;
		LoadManyToOneMap(value, srcWordToClusterId);
	} else if (key == "bitokenToBitokenId") {
		cvtBitokenToBitokenId = true;
		LoadManyToOneMap(value, bitokenToBitokenId);
	} else if (key == "bitokenIdToClusterId") {
		cvtBitokenIdToClusterId = true;
		LoadManyToOneMap(value, bitokenIdToClusterId);
	} else if (key == "lm") {
		m_lmPath = value;
	} else if (key == "ngrams") {
		nGramOrder = boost::lexical_cast<int>(value);
		//TODO: look at previous phrases
	} else {
		StatefulFeatureFunction::SetParameter(key, value);
	}
}

void CoarseBiLM::LoadManyToOneMap(const std::string& path,
		std::map<std::string, std::string> &manyToOneMap) {
	std::cerr << "LoadManyToOneMap Value: " + path << std::endl;

	std::ifstream file;
	file.open(path.c_str(), ios::in);

	string key;
	string value;

	while (file >> key >> value) {
		boost::algorithm::trim(key);
		boost::algorithm::trim(value);
		manyToOneMap[key] = value;
	}
	file.close();
}

void CoarseBiLM::readLanguageModel(const char *lmFile) {
	CoarseLM = ConstructCoarseLM(m_lmPath);

}

void CoarseBiLM::printList(const std::vector<std::string> &listToPrint) const {
	for (std::vector<std::string>::const_iterator iterator =
			listToPrint.begin(); iterator != listToPrint.end(); iterator++) {
		std::cerr << *iterator << " ";
	}
	std::cerr << endl;
}

std::string CoarseBiLM::getStringFromList(
		const std::vector<std::string> &listToConvert) const {
	std::string result = "";
	for (std::vector<std::string>::const_iterator iterator =
			listToConvert.begin(); iterator != listToConvert.end();
			iterator++) {
		result = result + " " + *iterator;
	}
	boost::algorithm::trim(result);
	return result;
}
}

