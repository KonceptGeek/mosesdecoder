#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "CoarseBiLM.h"
#include "moses/ScoreComponentCollection.h"
#include "moses/Hypothesis.h"
#include "moses/Manager.h"
#include "moses/InputType.h"
#include "moses/Word.h"

using namespace std;

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
	ReadParameters();
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
	vector<float> newScores(m_numScoreComponents);
	newScores[0] = 1.5;
	newScores[1] = 0.3;
	newScores[2] = 0.4;
	accumulator->PlusEquals(this, newScores);

	// sparse scores
	accumulator->PlusEquals(this, "sparse-name", 2.4);

	TargetPhrase currTargetPhrase = cur_hypo.GetCurrTargetPhrase();
	AlignmentInfo alignTerm = currTargetPhrase.GetAlignTerm();
	AlignmentInfo::CollType alignments = alignTerm.GetAlignments();


	std::size_t targetBegin = cur_hypo.GetCurrTargetWordsRange().GetStartPos();
	std::size_t targetEnd = cur_hypo.GetCurrTargetWordsRange().GetEndPos();
	std::size_t sourceBegin = cur_hypo.GetCurrSourceWordsRange().GetStartPos();
	std::size_t sourceEnd = cur_hypo.GetCurrSourceWordsRange().GetEndPos();

	vector<string> targetWords;
	if (targetBegin != targetEnd) {
		for (int index = targetBegin; index <= targetEnd; index++) {
			targetWords.push_back(cur_hypo.GetWord(index).ToString());
		}
	} else {
		targetWords.push_back(cur_hypo.GetWord(targetBegin).ToString());
	}

	Manager& manager = cur_hypo.GetManager();
	const InputType& sourceInput = manager.GetSource();

	vector<string> sourceWords;
	if (sourceBegin != sourceEnd) {
		for (int index = sourceBegin; index <= sourceEnd; index++) {
			sourceWords.push_back(sourceInput.GetWord(index).ToString());
		}
	} else {
		sourceWords.push_back(sourceInput.GetWord(sourceBegin).ToString());
	}

	std::cerr
			<< "Printing list, source: "
					+ boost::lexical_cast<std::string>(sourceWords.size())
			<< std::endl;

	std::cerr << "Printing list, source and then target: " << std::endl;
	for (vector<string>::const_iterator iterator = sourceWords.begin();
			iterator != sourceWords.end(); iterator++)
		std::cerr << " " << *iterator;
	std::cerr << endl;

	std::cerr
			<< "Printing list, target: "
					+ boost::lexical_cast<std::string>(targetWords.size())
			<< std::endl;
	for (vector<string>::const_iterator iterator = targetWords.begin();
			iterator != targetWords.end(); iterator++)
		std::cerr << " " << *iterator;
	std::cerr << endl;

	return new CoarseBiLMState(0);
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
		tgtWordToClusterId = LoadManyToOneMap(value);
	} else if (key == "srcWordToClusterId") {
		srcWordToClusterId = LoadManyToOneMap(value);
	} else if (key == "bitokenToBitokenId") {
		bitokenToBitokenId = LoadManyToOneMap(value);
	} else if (key == "bitokenIdToClusterId") {
		bitokenIdToClusterId = LoadManyToOneMap(value);
	} else if (key == "lm") {
		//TODO: load language model
	} else {
		StatefulFeatureFunction::SetParameter(key, value);
	}
}

std::map<std::string, std::string> CoarseBiLM::LoadManyToOneMap(
		const std::string& path) {
	std::cerr << "LoadManyToOneMap Value: " + path << std::endl;

	std::map<std::string, std::string> manyToOneMap;
	manyToOneMap["testKey"] = "testValue";
	return manyToOneMap;
}

}

