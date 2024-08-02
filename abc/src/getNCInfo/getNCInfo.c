#include "base/abc/abc.h"
#include "base/main/main.h"
#include "base/main/mainInt.h"
#include "proof/fraig/fraig.h"
#include "opt/fxu/fxu.h"
#include "opt/fxch/Fxch.h"
#include "opt/cut/cut.h"
#include "map/fpga/fpga.h"
#include "map/if/if.h"
#include "opt/sim/sim.h"
#include "opt/res/res.h"
#include "opt/lpk/lpk.h"
#include "aig/gia/giaAig.h"
#include "opt/dar/dar.h"
#include "opt/mfs/mfs.h"
#include "proof/fra/fra.h"
#include "aig/saig/saig.h"
#include "proof/int/int.h"
#include "proof/dch/dch.h"
#include "proof/ssw/ssw.h"
#include "opt/cgt/cgt.h"
#include "bool/kit/kit.h"
#include "map/amap/amap.h"
#include "opt/ret/retInt.h"
#include "sat/xsat/xsat.h"
#include "sat/satoko/satoko.h"
#include "sat/cnf/cnf.h"
#include "proof/cec/cec.h"
#include "proof/acec/acec.h"
#include "proof/pdr/pdr.h"
#include "misc/tim/tim.h"
#include "bdd/llb/llb.h"
#include "bdd/bbr/bbr.h"
#include "map/cov/cov.h"
#include "base/cmd/cmd.h"
#include "proof/abs/abs.h"
#include "sat/bmc/bmc.h"
#include "proof/ssc/ssc.h"
#include "opt/sfm/sfm.h"
#include "opt/sbd/sbd.h"
#include "bool/rpo/rpo.h"
#include "map/mpm/mpm.h"
#include "opt/fret/fretime.h"
#include "opt/nwk/nwkMerge.h"
#include "base/acb/acbPar.h"
#include "map/mapper/mapperInt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <unistd.h>
#endif

ABC_NAMESPACE_IMPL_START

////////////////////////////////////////////////////////////////////////
///                        DECLARATIONS                              ///
////////////////////////////////////////////////////////////////////////

int Abc_getNCInfo(Abc_Ntk_t* pNtk, char* nodesFile, char* cutsFile, double DelayTarget, double AreaMulti, double DelayMulti, float LogFan, float Slew, float Gain, int nGatesMin, int fRecovery, int fSwitching, int fSkipFanout, int fUseProfile, int fUseBuffs, int fVerbose);
int Map_DumpNCFeatures(Map_Man_t * pMan, char* nodesFile, char* cutsFile);
void Map_DumpCutFeatures( Map_Man_t * pMan, Map_Node_t * pNode, Map_Cut_t * pCut, FILE * cutsFile);

extern Vec_Int_t * Sim_NtkComputeSwitching( Abc_Ntk_t * pNtk, int nPatterns );
extern Map_Man_t *  Abc_NtkToMap( Abc_Ntk_t * pNtk, double DelayTarget, int fRecovery, float * pSwitching, int fVerbose );
extern int Map_MatchNodeCut( Map_Man_t * p, Map_Node_t * pNode, Map_Cut_t * pCut, int fPhase, float fWorstLimit );
extern int Map_MatchCompare( Map_Man_t * pMan, Map_Match_t * pM1, Map_Match_t * pM2, int fDoingArea );
extern void Map_MatchClean( Map_Match_t * pMatch );

////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////  


int Abc_getNCInfo(Abc_Ntk_t * pNtk, char* nodesFile, char* cutsFile, double DelayTarget, double AreaMulti, double DelayMulti, float LogFan, float Slew, float Gain, int nGatesMin, int fRecovery, int fSwitching, int fSkipFanout, int fUseProfile, int fUseBuffs, int fVerbose) 
{
    static int fUseMulti = 0;
    int fShowSwitching = 1;

    Abc_Ntk_t * pNtkNew;
    Map_Man_t * pMan;
    Vec_Int_t * vSwitching = NULL;
    float * pSwitching = NULL;
    abctime clk, clkTotal = Abc_Clock();
    Mio_Library_t * pLib = (Mio_Library_t *)Abc_FrameReadLibGen();

    assert( Abc_NtkIsStrash(pNtk) );
    // derive library from SCL
    // if the library is created here, it will be deleted when pSuperLib is deleted in Map_SuperLibFree()
    if ( Abc_FrameReadLibScl() && Abc_SclHasDelayInfo( Abc_FrameReadLibScl() ) )
    {
        if ( pLib && Mio_LibraryHasProfile(pLib) )
            pLib = Abc_SclDeriveGenlib( Abc_FrameReadLibScl(), pLib, Slew, Gain, nGatesMin, fVerbose );
        else
            pLib = Abc_SclDeriveGenlib( Abc_FrameReadLibScl(), NULL, Slew, Gain, nGatesMin, fVerbose );
        if ( Abc_FrameReadLibGen() )
        {
            Mio_LibraryTransferDelays( (Mio_Library_t *)Abc_FrameReadLibGen(), pLib );
            Mio_LibraryTransferProfile( pLib, (Mio_Library_t *)Abc_FrameReadLibGen() );
        }
        // remove supergate library
        Map_SuperLibFree( (Map_SuperLib_t *)Abc_FrameReadLibSuper() );
        Abc_FrameSetLibSuper( NULL );
    }
    // quit if there is no library
    if ( pLib == NULL )
    {
        printf( "The current library is not available.\n" );
        return 0;
    }
    
    // derive the supergate library
    if ( fUseMulti || Abc_FrameReadLibSuper() == NULL )
    {
        if ( fVerbose )
            printf( "Converting \"%s\" into supergate library \"%s\".\n", 
                Mio_LibraryReadName(pLib), Extra_FileNameGenericAppend(Mio_LibraryReadName(pLib), ".super") );
        // compute supergate library to be used for mapping
        if ( Mio_LibraryHasProfile(pLib) )
            printf( "Abc_NtkMap(): Genlib library has profile.\n" );
        Map_SuperLibDeriveFromGenlib( pLib, fVerbose );
    }
    // print a warning about choice nodes
    if ( fVerbose && Abc_NtkGetChoiceNum( pNtk ) )
        printf( "Performing mapping with choices.\n" );

    // compute switching activity
    fShowSwitching |= 0;
    if ( fShowSwitching )
    {
        vSwitching = Sim_NtkComputeSwitching( pNtk, 4096 );
        pSwitching = (float *)vSwitching->pArray;
    }

    // start the mapping manager and set its parameters
    pMan = Map_ManCreate( Abc_NtkPiNum(pNtk) + Abc_NtkLatchNum(pNtk) - pNtk->nBarBufs, Abc_NtkPoNum(pNtk) + Abc_NtkLatchNum(pNtk) - pNtk->nBarBufs, fVerbose);
    if ( pMan == NULL ){
        return NULL;
    }
    
    assert(pMan!=NULL);
    assert(pNtk!=NULL);
    pMan = Abc_NtkToMap( pNtk, -1, 1, NULL, 0 );
    Map_DumpNCFeatures( pMan, nodesFile, cutsFile); 
    Map_ManFree( pMan );
    return 1; 
}

int Map_DumpNCFeatures(Map_Man_t * pMan, char* nodesFile, char* cutsFile)
{
    int i = 0;
    Map_Node_t * pNode; 
    Map_MappingSetChoiceLevels( pMan ); // should always be called before mapping
    // compute the cuts of nodes in the DFS order
    Map_MappingCuts( pMan );
    Map_MappingTruths( pMan );

    FILE* cuts_file; 
    FILE* nodes_file; 

    cuts_file = fopen(cutsFile, "w+");
    nodes_file = fopen(nodesFile, "w+");

    fprintf(cuts_file, "root_idx,cut_idx,l1_idx,l2_idx,l3_idx,l4_idx,l5_idx,vol_cut,cut_height,canon_tt_0,cannon_tt_1\n");
    fprintf(nodes_file, "node_idx,type,node_inv,num_fo,lvl,rev_lvl,c1_idx,c2_idx\n");
    int totalLevel = Map_MappingGetMaxLevel(pMan); 
    int c1_idx,c2_idx,type;

    for ( i = 0; i < pMan->vMapObjs->nSize; i++ )
    {
        pNode = pMan->vMapObjs->pArray[i];
        if(Map_NodeIsConst(pNode)){
            type = -2;
            c1_idx = -1;
            c2_idx = -1;
        }else if(Map_NodeIsVar(pNode)){
            type = 0;
            c1_idx = -1;
            c2_idx = -1;
        }else if(Map_NodeIsBuf(pNode)){
            type = -1;
            c1_idx = Map_Regular(pNode->p1)->Num;
            c2_idx = -1;
        }else{
            type = 1;
            c1_idx = Map_Regular(pNode->p1)->Num;
            c2_idx = Map_Regular(pNode->p2)->Num;
        }

        for(int j = 0; j < pMan->nOutputs; j++){
            if(Map_Regular(pMan->pOutputs[j])->Num == pNode->Num) type = 2;
        }
        int rev_lvl = totalLevel - pNode->Level;
        fprintf(nodes_file, "%d,%d,%u,%d,%u,%d,%d,%d\n", pNode->Num,type, pNode->fInv, pNode->nRefs, pNode->Level, rev_lvl, c1_idx, c2_idx);
        Map_Cut_t * pCut;

        if ( pNode->pCuts->pNext == NULL ) continue;

        for ( pCut = pNode->pCuts->pNext; pCut; pCut = pCut->pNext )
        {
            Map_DumpCutFeatures( pMan, pNode, pCut, cuts_file); 
        }
    }

    if (cuts_file != "")
        fclose(cuts_file);
    if (nodes_file != "")
        fclose(nodes_file);

    return 1;  
}

void Map_DumpCutFeatures( Map_Man_t * p, Map_Node_t * pNode, Map_Cut_t * pCut, FILE * cutsFile)
{
    Map_Node_t * pNodeR;
    int minLevel = 1000000, cut_height = 0, i;
    int leavesIdx[5] = { -1, -1, -1, -1, -1 };
    pNodeR = Map_Regular(pNode);
    for ( i = 0; i < p->nVarsMax; i++ ) {
        if ( pCut->ppLeaves[i] ) { 
            if (pCut->ppLeaves[i]->Level < minLevel)
                minLevel = pCut->ppLeaves[i]->Level; 
            leavesIdx[i] = pCut->ppLeaves[i]->Num; 
        }
    }
    cut_height = pNodeR->Level - minLevel; 
    fprintf(cutsFile, "%d,%lld,%d,%d,%d,%d,%d,%d,%d,%u,%u\n", pNodeR->Num, pCut->cutIdx, leavesIdx[0], leavesIdx[1], leavesIdx[2], leavesIdx[3], leavesIdx[4], pCut->nVolume, cut_height, pCut->uCanon[0][0], pCut->uCanon[1][0]);
} 

ABC_NAMESPACE_IMPL_END