#ifndef ABC__SelectiveMap_h
#define ABC__SelectiveMap_h

#include "base/main/main.h"

ABC_NAMESPACE_HEADER_START

////////////////////////////////////////////////////////////////////////
///                        DECLARATIONS                              ///
////////////////////////////////////////////////////////////////////////

/*=== selectiveMap.c =============================================================*/
Abc_Ntk_t * Abc_selectiveMap ( Abc_Ntk_t * pNtk, char * cutChoiceFile, char* dumpFile , double DelayTarget, double AreaMulti, double DelayMulti, float LogFan, float Slew, float Gain, int nGatesMin, int fRecovery, int fSwitching, int fSkipFanout, int fUseProfile, int fUseBuffs, int fVerbose);

#endif
ABC_NAMESPACE_HEADER_END