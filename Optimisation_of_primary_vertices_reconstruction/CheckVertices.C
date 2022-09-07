#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <stddef.h>
#include <algorithm>

#include <TFile.h>
#include <TTree.h>
#include <TClonesArray.h>
#include <TH2F.h>
#include <TCanvas.h>
#include <THStack.h>
#include <TLegend.h>
#include <TPad.h>
#include <TGraph.h>
#include <TMultiGraph.h>
#include <TMath.h>
#include <TStyle.h>

#include "ITSBase/GeometryTGeo.h"
#include "SimulationDataFormat/TrackReference.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITS/TrackITS.h"

#include "ReconstructionDataFormats/Vertex.h"

#include "DataFormatsITSMFT/ROFRecord.h"

#endif

#pragma link C++ class ParticleInfo + ;

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;
using namespace o2::itsmft;
using namespace o2::its;

struct ParticleInfo
{
  int event;
  int pdg;
  float pt;
  float eta;
  float phi;
  int mother;
  int first;
  unsigned short clusters = 0u;
  unsigned char isReco = 0u;
  unsigned char isFake = 0u;
  bool isPrimary = 0u;
  unsigned char storedStatus = 2; /// not stored = 2, fake = 1, good = 0
  o2::its::TrackITS track;
};

struct SimVertexInfo
{
  unsigned int EvId;
  float Vx;
  float Vy;
  float Vz;
  // unsigned int NumTr;//without condition part.clusters >= 0x7
  unsigned int NumTrIB_rec = 0u;
  unsigned int NumTrIB_mc = 0u;
};

struct RecVertexInfo
{
  float Vx;
  float Vy;
  float Vz;
  unsigned int NumContr = 0u;
  std::map<unsigned int, unsigned int> LabelMap;
  int GetTopLabel() const
  {
    if (LabelMap.size() == 0)
      return -1;
    std::vector<std::pair<unsigned int, unsigned int>> vec;
    vec.reserve(LabelMap.size());
    for (auto &i : LabelMap)
      vec.push_back(i);
    std::sort(vec.begin(), vec.end(), [=](std::pair<unsigned int, unsigned int> &left, std::pair<unsigned int, unsigned int> &right)
              { return left.second > right.second; });
    return vec[0].first;
  }
  double Purity() // purity : % of top label
  {
    if (GetTopLabel() == -1)
      return 0;
    int tot_num_labels = 0;
    for (const auto &[key, value] : LabelMap)
      tot_num_labels += value;
    return (double)LabelMap[GetTopLabel()] / tot_num_labels;
  }
};

Int_t getHadronQ(Int_t id_num)
{
  Int_t Q = 0;
  Int_t digits[4] = {0};
  Int_t charge[6] = {-1, 2, -1, 2, -1, 2};
  Int_t anti = 1;
  if (id_num < 0)
    anti = -1;
  Int_t id = TMath::Abs(id_num);
  // adding to arr last 4 num of code (we`ll use only n-3 .. n-1 nums)
  for (Int_t i = 0; i < 4; ++i)
  {
    digits[i] = (id / Int_t((TMath::Power(10, i)))) % 10;
  }
  // if it`s not a hadron return 0
  if (id < 100 || (id > 1000 && digits[1] == 0))
  {
    return Q;
  }
  // baryon
  if (digits[3] != 0)
  {
    for (Int_t i = 1; i < 4; ++i)
    {
      Q += charge[digits[i] - 1];
    }
  }
  // meson

  else
  {
    /*if (digits[2] % 2 != 0)
      {
          Q += charge[digits[1] - 1];
          Q -= charge[digits[2] - 1];
      }
      else
      {
          Q -= charge[digits[1] - 1];
          Q += charge[digits[2] - 1];
      }*/
    Q += charge[digits[1] - 1];
    Q -= charge[digits[2] - 1];
    Q = TMath::Abs(Q);
  }
  Q *= anti;
  Q /= 3;
  return Q;
}

bool isCharged(TDatabasePDG *pdg, Int_t part_pdg_code)
{ // if the particle is charged return true
  if ((pdg->GetParticle(part_pdg_code) == nullptr) && (((int)part_pdg_code / 1000) != 0))
    return ((getHadronQ(part_pdg_code) == 0) ? false : true);
  if ((pdg->GetParticle(part_pdg_code) == nullptr))
    return false;
  return ((int)TMath::Abs(pdg->GetParticle(part_pdg_code)->Charge()) != 0);
}

//============================================================================================================================
void CheckVertices(std::string tracfile = "o2trac_its.root", std::string clusfile = "o2clus_its.root", std::string kinefile = "o2sim_Kine.root")
{
  // Geometry
  o2::base::GeometryManager::loadGeometry();
  auto gman = o2::its::GeometryTGeo::Instance();

  // MC tracks

  TFile *file0 = TFile::Open(kinefile.data());
  TTree *mcTree = (TTree *)file0->Get("o2sim");

  mcTree->SetBranchStatus("*", 0);        // disable all branches
  mcTree->SetBranchStatus("MCTrack*", 1); /// the star at the end must be here!

  std::vector<o2::MCTrack> *mcArr = nullptr;
  mcTree->SetBranchAddress("MCTrack", &mcArr);

  // Clusters
  TFile::Open(clusfile.data());
  TTree *clusTree = (TTree *)gFile->Get("o2sim");

  std::vector<CompClusterExt> *clusArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterComp", &clusArr);

  o2::dataformats::MCTruthContainer<o2::MCCompLabel> *clusLabArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);

  // Reconstructed tracks
  TFile *file1 = TFile::Open(tracfile.data());
  TTree *recTree = (TTree *)file1->Get("o2sim");

  std::vector<Vertex> *recVerArr = nullptr;
  recTree->SetBranchAddress("Vertices", &recVerArr);

  std::vector<ROFRecord> *recVerROFArr = nullptr;
  recTree->SetBranchAddress("VerticesROF", &recVerROFArr);

  std::vector<o2::MCCompLabel> *recLabelsArr = nullptr;
  recTree->SetBranchAddress("ITSVertexMCTruth", &recLabelsArr);

  std::vector<TrackITS> *recArr = nullptr;
  recTree->SetBranchAddress("ITSTrack", &recArr);

  std::vector<o2::MCCompLabel> *trkLabArr = nullptr;
  recTree->SetBranchAddress("ITSTrackMCTruth", &trkLabArr);

  //============================================================================================================================

  // std::cout << "** Filling particle table ... " << std::flush;
  int lastEventIDcl = -1, cf = 0;
  int nev = mcTree->GetEntriesFast();
  std::vector<std::vector<ParticleInfo>> info(nev);
  for (int n = 0; n < nev; n++)
  { // loop over MC events
    mcTree->GetEvent(n);
    info[n].resize(mcArr->size());
    for (unsigned int mcI{0}; mcI < mcArr->size(); ++mcI)
    {
      auto part = mcArr->at(mcI);
      info[n][mcI].event = n;
      info[n][mcI].pdg = part.GetPdgCode();
      info[n][mcI].pt = part.GetPt();
      info[n][mcI].phi = part.GetPhi();
      info[n][mcI].eta = part.GetEta();
      info[n][mcI].isPrimary = part.isPrimary();
    }
  }

  // std::cout << "** Creating particle/clusters correspondance ... " << std::flush;
  for (int frame = 0; frame < clusTree->GetEntriesFast(); frame++)
  { // Cluster frames
    if (!clusTree->GetEvent(frame))
      continue;

    for (unsigned int iClus{0}; iClus < clusArr->size(); ++iClus)
    {
      auto lab = (clusLabArr->getLabels(iClus))[0];
      if (!lab.isValid() || lab.getSourceID() != 0 || !lab.isCorrect())
        continue;

      int trackID, evID, srcID;
      bool fake;
      lab.get(trackID, evID, srcID, fake);
      if (evID < 0 || evID >= (int)info.size())
      {
        std::cout << "Cluster MC label eventID out of range" << std::endl;
        continue;
      }
      if (trackID < 0 || trackID >= (int)info[evID].size())
      {
        std::cout << "Cluster MC label trackID out of range" << std::endl;
        continue;
      }

      const CompClusterExt &c = (*clusArr)[iClus];
      auto layer = gman->getLayer(c.getSensorID());
      info[evID][trackID].clusters |= 1 << layer;
    }
  }

  // std::cout << "** Analysing tracks ... " << std::flush;
  int unaccounted{0}, good{0}, fakes{0}, total{0};
  for (int frame = 0; frame < recTree->GetEntriesFast(); frame++)
  { // Cluster frames
    if (!recTree->GetEvent(frame))
      continue;
    total += trkLabArr->size();
    for (unsigned int iTrack{0}; iTrack < trkLabArr->size(); ++iTrack)
    {
      auto lab = trkLabArr->at(iTrack);
      if (!lab.isSet())
      {
        unaccounted++;
        continue;
      }
      int trackID, evID, srcID;
      bool fake;
      lab.get(trackID, evID, srcID, fake);
      if (evID < 0 || evID >= (int)info.size())
      {
        unaccounted++;
        continue;
      }
      if (trackID < 0 || trackID >= (int)info[evID].size())
      {
        unaccounted++;
        continue;
      }
      info[evID][trackID].isReco += !fake;
      info[evID][trackID].isFake += fake;
      /// We keep the best track we would keep in the data
      if (recArr->at(iTrack).isBetter(info[evID][trackID].track, 1.e9))
      {
        info[evID][trackID].storedStatus = fake;
        info[evID][trackID].track = recArr->at(iTrack);
      }

      fakes += fake;
      good += !fake;
    }
  }

  //============================================================================================================================
  //============================================================================================================================
  //============================================================================================================================
  //============================================================================================================================
  //============================================================================================================================

  // Creating a particle database manager class (for PDG code)
  TDatabasePDG *pdg = new TDatabasePDG();

  // MC tracks

  unsigned int num_ev = mcTree->GetEntries(); // how many times was filled the tree (events)
  std::vector<SimVertexInfo> mc_info(num_ev);
  // loop over MC events
  for (unsigned int n = 0; n < num_ev; ++n)
  {
    mcTree->GetEvent(n); // filling the mcArr for event n
    mc_info[n].EvId = n;
    bool mc_find_ver = true;
    for (unsigned int i{0}; i < mcArr->size(); ++i)
    {
      auto mc_part = mcArr->at(i); // operator [] ====> ->at()
      // particle has to be: primary, charged, eta < 1.1, pt > pt(min) = q |B| Rc(min) = 10^-2 * 2.1 GeV/c
      if (mc_part.isPrimary() && TMath::Abs(mc_part.GetEta() < 1.1) && mc_part.GetPt() > 0.02105 && isCharged(pdg, mc_part.GetPdgCode()))
      {
        // lets write the first found vertix
        if (mc_find_ver)
        {
          mc_info[n].Vx = mc_part.GetStartVertexCoordinatesX();
          mc_info[n].Vy = mc_part.GetStartVertexCoordinatesY();
          mc_info[n].Vz = mc_part.GetStartVertexCoordinatesZ();
          mc_find_ver = false;
        }
        // mc_info[n].NumTr += 1;
      }
    }
  }

  // Some interesting statistic for simulated data
  /*
  ////loop over all particles
  Ntot = 53703504 (all particles in all events)
  Nnor = 10792048 -> 20.0956% (particles that easy to find in the scheme)
  Nstr = 42911456 -> 79.9044% (all the rest)
  Nhadr = 42910983 -> 79.9035% (particle codes that longer than 4 digits)
  Nnull = 473 -> 0.000880762% (code has less than 4 digits and missing from the table)
  N_990 = 473 (num of particles with code 990 (pomeron))
  ///////loop over all primary particles
  Ntot = 1077886
  Nnor = 1076609 -> 99.8815%
  Nstr = 1277 -> 0.118473%
  Nhadr = 804 -> 0.0745904%
  Nnull = 473 -> 0.0438822%
  N_990 = 473
  ///////loop over primary particles with eta < 1.1, pt < pt_min
  Ntot = 593131
  Nnor = 592467 -> 99.8881%
  Nstr = 664 -> 0.111948%
  Nhadr = 390 -> 0.0657528%
  Nnull = 274 -> 0.0461955%
  N_990 = 274
  */

  //============================================================================================================================
  // Reconstructed tracks
  // Filling the array with eventIDs
  std::vector<int> eventIdArr;
  for (int frame = 0; frame < recTree->GetEntriesFast(); ++frame)
  {
    recTree->GetEvent(frame);
    for (unsigned int tracklet_i = 0; tracklet_i < recLabelsArr->size(); ++tracklet_i)
    {
      auto rec_tr = recLabelsArr->at(tracklet_i);
      if (rec_tr.getEventID() == 524287)
        eventIdArr.push_back(-1);
      else
        eventIdArr.push_back(rec_tr.getEventID());
    }
  }

  // Filling the rec_info array
  std::vector<RecVertexInfo> rec_info;
  unsigned int point = 0;
  /*
    rec_info.resize(rec_info.size() + recVerArr->size());
    for (int readout_frame = 0; readout_frame < recVerROFArr->size(); ++readout_frame)
    {
      for (int i_ver = (*recVerROFArr)[readout_frame].getFirstEntry(); i_ver < (*recVerROFArr)[readout_frame].getFirstEntry() + (*recVerROFArr)[readout_frame].getNEntries(); ++i_ver)
      {
        auto rec_ver = (*recVerArr)[i_ver]; // auto rec_ver = recVerArr->at(i_ver);
        rec_info[i_ver].Vx = rec_ver.getX();
        rec_info[i_ver].Vy = rec_ver.getY();
        rec_info[i_ver].Vz = rec_ver.getZ();
        rec_info[i_ver].NumContr = rec_ver.getNContributors();
        vector<int> temp_labels;
        for (unsigned int j = 0; j < rec_ver.getNContributors(); ++j)
          temp_labels.push_back(eventIdArr[j + point]);
        point += rec_ver.getNContributors();
        // std::sort(std::begin(temp_labels), std::end(temp_labels));
        for (unsigned int j = 0; j < temp_labels.size(); ++j)
        {
          if (temp_labels[j] == -1)
            continue;
          ++rec_info[i_ver].LabelMap[temp_labels[j]];
        }
      }
    }*/
  for (int frame = 0; frame < recTree->GetEntries(); ++frame)
  {
    recTree->GetEvent(frame);
    rec_info.resize(rec_info.size() + recVerArr->size());
    for (unsigned int i_ver{0}; i_ver < recVerArr->size(); ++i_ver)
    {
      auto rec_ver = recVerArr->at(i_ver);
      rec_info[i_ver].Vx = rec_ver.getX();
      rec_info[i_ver].Vy = rec_ver.getY();
      rec_info[i_ver].Vz = rec_ver.getZ();
      rec_info[i_ver].NumContr = rec_ver.getNContributors();
      vector<int> temp_labels;
      for (unsigned int j = 0; j < rec_ver.getNContributors(); ++j)
        temp_labels.push_back(eventIdArr[j + point]);
      point += rec_ver.getNContributors();
      // std::sort(std::begin(temp_labels), std::end(temp_labels));
      for (unsigned int j = 0; j < temp_labels.size(); ++j)
      {
        if (temp_labels[j] == -1)
          continue;
        ++rec_info[i_ver].LabelMap[temp_labels[j]];
      }
    }
  }
  for (unsigned int i = 0; i < info.size(); ++i)
  {
    for (unsigned int j = 0; j < info[i].size(); ++j)
    {
      auto part = info[i][j];
      if (part.clusters >= 0x7 && part.isPrimary && TMath::Abs(part.eta) < 1.96 && part.pt > 0.02105 && isCharged(pdg, part.pdg) && part.isReco)
        mc_info[i].NumTrIB_rec += 1;
      if (part.clusters >= 0x7 && part.isPrimary && TMath::Abs(part.eta) < 1.96 && part.pt > 0.02105 && isCharged(pdg, part.pdg))
        mc_info[i].NumTrIB_mc += 1;
    }
  }

  // NumContr vs NumRec
  /*
    for (int i = 0; i < rec_info.size(); ++i)
    {
      if (rec_info[i].GetTopLabel() == -1)
        std::cout << "false : " << "ev # " << rec_info[i].GetTopLabel() << "\tNum_contr : " << rec_info[i].NumContr << "\n";
      else
        std::cout << "ev # " << rec_info[i].GetTopLabel() << "\tNum_contr : " << rec_info[i].NumContr << "\tNum_rec : " << mc_info[rec_info[i].GetTopLabel()].NumTrIB_rec << "\n";
    }*/

  // Printing full information about the events
  /*
  // for (auto &elem : eventIdArr)
  //   std::cout << elem << "\t";
  // std::cout << "\n\n\n";
  for (unsigned int n = 0; n < N_total; ++n)
  {
    std::cout << "=================================== Ev #" << n << " =================================== \n";
    std::cout << "\tNumrec = " << mc_info[n].NumTrIB_rec << "\tNumsim = " << mc_info[n].NumTrIB_mc << "\n";
    unsigned int counter = 0u;
    for (unsigned int i = 0; i < rec_info.size(); ++i)
    {
      if (rec_info[i].GetTopLabel() == (int)n)
        counter += 1;
    }
    if (counter == 0)
      std::cout << "------------------FALSE------------------\n";
    else
    {
      for (unsigned int i = 0; i < rec_info.size(); ++i)
      {
        if (rec_info[i].GetTopLabel() == (int)n)
        {
          if (rec_info[i].Purity() < 1)
          {
            std::cout << "------------------TRUE------------------\n";
            std::cout << "\tNumContr = " << rec_info[i].NumContr << "\tPurity = " << rec_info[i].Purity() << std::endl;
            std::cout << "Possible vertex --> Label Map\n";
            for (auto &el : rec_info[i].LabelMap)
              std::cout << "\t\t\t" << el.first << " " << el.second << std::endl;
          }
        }
      }
    }
  }
  std::cout << "\n\n\n\n";*/

  // Printing the efficiency
  unsigned int N_dublicated = 0u, N_untrue = 0u, N_successful = 0u;
  const unsigned int N_total = num_ev;
  for (unsigned int n = 0; n < N_total; ++n)
  {
    unsigned int counter = 0u;
    for (unsigned int i = 0; i < rec_info.size(); ++i)
    {
      if (rec_info[i].GetTopLabel() == (int)n)
        counter += 1;
    }
    if (counter == 1)
      N_successful += 1;
    else if (counter > 1)
      N_dublicated += 1;
    else
      N_untrue += 1;
  }

  // Vertices with purity < 100 %, that contains two kinds of labels that are equally suitable for 2 events? Lets check it
  // How many vertices could we describe with labels from other vertices?
  std::vector<unsigned int> suspicious_labels;
  for (unsigned int i = 0; i < rec_info.size(); ++i)
  {
    if (rec_info[i].Purity() < 100)
    {
      for (const auto &[key, value] : rec_info[i].LabelMap)
      {
        if ((int)key != rec_info[i].GetTopLabel())
          suspicious_labels.push_back(key);
      }
    }
  }
  unsigned int N_skipped = 0u;
  for (unsigned int n = 0; n < N_total; ++n)
  {
    unsigned int counter = 0u;
    for (unsigned int i = 0; i < rec_info.size(); ++i)
    {
      if (rec_info[i].GetTopLabel() == (int)n)
        counter += 1;
    }
    if (counter == 0)
    {
      for (auto &elem : suspicious_labels)
      {
        if (elem == n)
          N_skipped += 1;
      }
    }
  }

  // Calculation of the purity and efficiency
  double mPurity = 0;
  int mEfficiency = 0;
  for (unsigned int n = 0; n < N_total; ++n)
  {
    std::vector<unsigned int> temp; //indices arr
    for (unsigned int i = 0; i < rec_info.size(); ++i)
    {
      if (rec_info[i].GetTopLabel() == (int)n)
        temp.push_back(i);
    }
    if (temp.size() > 0)
    {
      double max_pur = 0;
      for (unsigned int i = 0; i < temp.size(); ++i)
      {
        double temp_pur = rec_info[temp[i]].Purity();
        if (temp_pur > max_pur)
          max_pur = temp_pur;
      }
      mPurity += max_pur;
      mEfficiency +=1;
    }
  }

  // Calculation the number of reconstructed vertices with Purity > 0.7

  unsigned int N_pur_more_pers = 0u, N_pur_less_pers = 0u;
  for (unsigned int n = 0; n < N_total; ++n)
  {
    unsigned int counter = 0u;
    int temp_index = -1;
    for (unsigned int i = 0; i < rec_info.size(); ++i)
    {
      if (rec_info[i].GetTopLabel() == (int)n)
      {
        counter += 1;
        temp_index = i;
      }
    }
    if (counter == 1 && temp_index >= 0)
    {
      if (rec_info[temp_index].Purity() >= 0.7)
        N_pur_more_pers += 1;
      else
        N_pur_less_pers += 1;
    }
  }

  std::cout << "\n--------------------------------------Results--------------------------------------\n";

  std::cout << "\nN_total = " << N_total
            << "\nN_successful = " << N_successful
            << "\nN_dublicated = " << N_dublicated
            << "\nN_untrue = " << N_untrue
            << "\nN_skipped = " << N_skipped
            << "\nN_vertices_with_purity>=0.7 = " << N_pur_more_pers
            << "\nN_vertices_with_purity<0.7 = " << N_pur_less_pers << "\n";

  std::cout << "\nEfficiency for vertices : " << (double)mEfficiency /  N_total
            << "\nPersentage of dublicated vertices : " << (double)N_dublicated / N_total
            << "\nPersentage of untrue vertices : " << (double)N_untrue / N_total
            << "\nPersentage of skipped vertices : " << (double)N_skipped / N_total
            << "\nPersentage of vertices with purity >= 0.7  : " << (double)N_pur_more_pers / N_total
            << "\nPersentage of vertices with purity < 0.7 : " << (double)N_pur_less_pers / N_total << "\n";

  std::cout << "Mean Purity : " << mPurity / N_total << "\n";

  //============================================================================================================================
  //============================================================================================================================
  //============================================================================================================================
  //============================================================================================================================
  //============================================================================================================================

  //====================================================PLOTS===================================================================

  /*
  int num_bins_pos = 200;
  double x_max = 2.1, x_min = -x_max;
  double y_max = 0.1, y_min = -y_max;
  double z_max = 20, z_min = -z_max;
  TH1D *verX = new TH1D("verX", " ; #it{x}, cm ; ", num_bins_pos, x_min, x_max);
  verX->Sumw2();
  TH1D *verY = new TH1D("verY", " ; #it{y}, cm ; ", num_bins_pos, y_min, y_max);
  verY->Sumw2();
  TH1D *verZ = new TH1D("verZ", " ; #it{z}, cm ; ", num_bins_pos, z_min, z_max);
  verZ->Sumw2();

  for (auto &mc_ev : mc_info)
  {
    verX->Fill(mc_ev.Vx);
    verY->Fill(mc_ev.Vy);
    verZ->Fill(mc_ev.Vz);
  }

  TCanvas *c1 = new TCanvas;
  c1->cd();
  c1->SetGridx();
  c1->SetGridy();
  TH1 *hist_Vi = (TH1 *)verX->Clone("hist_Vi");
  hist_Vi->SetLineColor(kBlack);
  hist_Vi->SetLineWidth(2);
  hist_Vi->SetFillColorAlpha(15, 0.1);
  hist_Vi->SetFillStyle(3244);

  hist_Vi->GetYaxis()->SetLabelFont(43);
  hist_Vi->GetYaxis()->SetLabelSize(26);
  hist_Vi->GetYaxis()->SetTitleFont(43);
  hist_Vi->GetYaxis()->SetTitleSize(20);
  hist_Vi->GetYaxis()->SetTitleOffset(2.0);

  hist_Vi->GetXaxis()->SetLabelFont(43);
  hist_Vi->GetXaxis()->SetLabelSize(26);
  hist_Vi->GetXaxis()->SetTitleFont(43);
  hist_Vi->GetXaxis()->SetTitleSize(20);
  hist_Vi->GetXaxis()->SetTitleOffset(1.1);
  // hist_Vx->GetYaxis()->SetRangeUser(0.,hist_Vx->GetMaximum()+10.);
  hist_Vi->Draw("hist");
  c1->SaveAs("Vx_coord_Sim.png", "recreate");

  // Plotting purity
  int num_bins_pur = 50;
  double x_axis_max = 101., x_axis_min = -1.;
  TH1D *hist_purity = new TH1D("Purity", " ; #it{Purity}, %; Number of vertices", num_bins_pur, x_axis_min, x_axis_max);
  hist_purity->Sumw2();

  for (auto &rec_ver : rec_info)
  {
    hist_purity->Fill(rec_ver.Purity());
  }

  // Plot of purity vs number of vertices with that purity
  TCanvas *c4 = new TCanvas;
  c4->cd();
  c4->SetGridx();
  c4->SetGridy();
  // gPad->SetTicks();
  //  gPad->SetTopMargin(0.05);
  //  gPad->SetLeftMargin(0.11);
  //  gPad->SetRightMargin(0.05);
  //  gPad->SetBottomMargin(0.1);
  //  gStyle->SetOptStat(0);
  //  gStyle->SetOptTitle(0);
  TLegend *legend_purity = new TLegend(0.5, 0.7, 0.98, 0.92);
  legend_purity->SetBorderSize(0);
  legend_purity->SetTextFont(43);
  legend_purity->SetTextSize(22);
  legend_purity->SetFillStyle(0);
  legend_purity->SetMargin(0.35);

  hist_purity->SetLineColor(kBlack);
  hist_purity->SetLineWidth(2);
  hist_purity->SetFillColorAlpha(15, 0.1);
  hist_purity->SetFillStyle(3244);
  // hist_purity->SetTitle("Purity");

  hist_purity->GetYaxis()->SetLabelFont(43);
  hist_purity->GetYaxis()->SetLabelSize(18);
  hist_purity->GetYaxis()->SetTitleFont(43);
  hist_purity->GetYaxis()->SetTitleSize(20);
  hist_purity->GetYaxis()->SetTitleOffset(2.0);

  hist_purity->GetXaxis()->SetLabelFont(43);
  hist_purity->GetXaxis()->SetLabelSize(18);
  hist_purity->GetXaxis()->SetTitleFont(43);
  hist_purity->GetXaxis()->SetTitleSize(20);
  hist_purity->GetXaxis()->SetTitleOffset(1.1);

  hist_purity->Draw("hist");
  legend_purity->Draw();
  c4->SaveAs("Purity.png", "recreate");

  // Plotting purity vs number of contributors

  unsigned int n_pc = rec_info.size();
  double x_pc[n_pc], y_pc[n_pc];
  for (unsigned int i = 0; i < n_pc; ++i)
  {
    y_pc[i] = rec_info[i].Purity();
    x_pc[i] = rec_info[i].NumContr;
  }
  TGraph *graph_purity_vs_numcontr = new TGraph(n_pc, x_pc, y_pc);

  TCanvas *c5 = new TCanvas;
  c5->cd();
  c5->SetGridx();
  c5->SetGridy();

  graph_purity_vs_numcontr->SetMarkerStyle(20);
  graph_purity_vs_numcontr->SetMarkerSize(1);
  graph_purity_vs_numcontr->SetMarkerColor(kBlack);

  graph_purity_vs_numcontr->GetYaxis()->SetLabelFont(43);
  graph_purity_vs_numcontr->GetYaxis()->SetLabelSize(18);
  graph_purity_vs_numcontr->GetYaxis()->SetTitleFont(43);
  graph_purity_vs_numcontr->GetYaxis()->SetTitleSize(15);
  graph_purity_vs_numcontr->GetYaxis()->SetTitleOffset(2.0);

  graph_purity_vs_numcontr->GetXaxis()->SetLabelFont(43);
  graph_purity_vs_numcontr->GetXaxis()->SetLabelSize(18);
  graph_purity_vs_numcontr->GetXaxis()->SetTitleFont(43);
  graph_purity_vs_numcontr->GetXaxis()->SetTitleSize(15);
  graph_purity_vs_numcontr->GetXaxis()->SetTitleOffset(1.1);

  graph_purity_vs_numcontr->GetYaxis()->SetTitle("#it{Purity}, %");
  graph_purity_vs_numcontr->GetXaxis()->SetTitle("#it{N}_{#it{contr}}");

  graph_purity_vs_numcontr->Draw("AP");
  c5->SaveAs("Purity_vs_NumContr.png", "recreate");

  // Plotting purity vs number of contributors

  std::vector<int> x_mult, y_mult;
  std::vector<int> x_mult_copy, y_mult_copy;
  std::vector<int> x_mult_false, y_mult_false;

  for (unsigned int n = 0; n < N_total; ++n)
  {
    unsigned int counter = 0u;
    for (unsigned int i = 0; i < rec_info.size(); ++i)
    {
      if (rec_info[i].GetTopLabel() == (int)n)
        counter += 1;
    }
    if (counter == 1)
    {
      x_mult.push_back((int)mc_info[n].NumTrIB_rec);
      y_mult.push_back((int)mc_info[n].NumTrIB_mc);
      // std::cout << (int)mc_info[n].NumTrIB_rec << "\t" << (int)mc_info[n].NumTrIB_mc << "\n";
    }
    if (counter > 1)
    {
      x_mult_copy.push_back((int)mc_info[n].NumTrIB_rec);
      y_mult_copy.push_back((int)mc_info[n].NumTrIB_mc);
    }
    if (counter == 0)
    {
      x_mult_false.push_back((int)mc_info[n].NumTrIB_rec);
      y_mult_false.push_back((int)mc_info[n].NumTrIB_mc);
    }
  }

  TMultiGraph *mg_mult = new TMultiGraph();
  TGraph *graph_mult = new TGraph(x_mult.size(), &x_mult[0], &y_mult[0]);
  TGraph *graph_mult_copy = new TGraph(x_mult_copy.size(), &x_mult_copy[0], &y_mult_copy[0]);
  TGraph *graph_mult_false = new TGraph(x_mult_false.size(), &x_mult_false[0], &y_mult_false[0]);

  TCanvas *c7 = new TCanvas;
  c7->cd();
  c7->SetGridx();
  c7->SetGridy();

  TLegend *legend = new TLegend(0.1, 0.7, 0.48, 0.9); //(0.5, 0.7, 0.98, 0.92);
  legend->SetBorderSize(0);
  legend->SetTextFont(43);
  legend->SetTextSize(18);
  legend->SetFillStyle(0);

  graph_mult->SetMarkerStyle(20);
  graph_mult->SetMarkerSize(0.8);
  graph_mult->SetMarkerColor(kBlack);

  graph_mult_copy->SetMarkerStyle(20);
  graph_mult_copy->SetMarkerSize(0.8);
  graph_mult_copy->SetMarkerColor(kBlue + 1);

  graph_mult_false->SetMarkerStyle(20);
  graph_mult_false->SetMarkerSize(0.8);
  graph_mult_false->SetMarkerColor(kRed + 1);

  mg_mult->Add(graph_mult);
  mg_mult->Add(graph_mult_copy);
  mg_mult->Add(graph_mult_false);

  mg_mult->GetYaxis()->SetLabelFont(43);
  mg_mult->GetYaxis()->SetLabelSize(18);
  mg_mult->GetYaxis()->SetTitleFont(43);
  mg_mult->GetYaxis()->SetTitleSize(15);
  mg_mult->GetYaxis()->SetTitleOffset(2.0);

  mg_mult->GetXaxis()->SetLabelFont(43);
  mg_mult->GetXaxis()->SetLabelSize(18);
  mg_mult->GetXaxis()->SetTitleFont(43);
  mg_mult->GetXaxis()->SetTitleSize(15);
  mg_mult->GetXaxis()->SetTitleOffset(1.1);

  mg_mult->GetYaxis()->SetTitle("#it{Multiplicity}_{#it{sim}}");
  mg_mult->GetXaxis()->SetTitle("#it{Multiplicity}_{#it{rec}}");

  legend->AddEntry(graph_mult, "Reconstructed vertices", "p");
  legend->AddEntry(graph_mult_copy, "Dublicated vertices", "p");
  legend->AddEntry(graph_mult_false, "Untrue vertices", "p");

  mg_mult->Draw("AP");
  legend->Draw();
  c7->SaveAs("NumRec_vs_NumMC.png", "recreate");

  // Plotting the efficiency to the Num_sim
  std::vector<int> x_mult_eff, y_mult_eff;

  for (unsigned int n = 0; n < N_total; ++n)
  {
    unsigned int counter = 0u;
    for (unsigned int i = 0; i < rec_info.size(); ++i)
    {
      if (rec_info[i].GetTopLabel() == (int)n)
        counter += 1;
    }
    x_mult_eff.push_back((int)mc_info[n].NumTrIB_mc);
    y_mult_eff.push_back((int)counter);
  }

  TGraph *graph_mult_eff = new TGraph(x_mult_eff.size(), &x_mult_eff[0], &y_mult_eff[0]);
  // TGraph *graph_mult_eff = new TGraph();
  TCanvas *c8 = new TCanvas;
  c8->cd();
  c8->SetGridx();
  c8->SetGridy();

  graph_mult_eff->SetTitle("");

  graph_mult_eff->SetMarkerStyle(20);
  graph_mult_eff->SetMarkerSize(0.8);
  graph_mult_eff->SetMarkerColor(kRed + 2);

  graph_mult_eff->GetYaxis()->SetLabelFont(43);
  graph_mult_eff->GetYaxis()->SetLabelSize(18);
  graph_mult_eff->GetYaxis()->SetTitleFont(43);
  graph_mult_eff->GetYaxis()->SetTitleSize(15);
  graph_mult_eff->GetYaxis()->SetTitleOffset(2.0);

  graph_mult_eff->GetXaxis()->SetLabelFont(43);
  graph_mult_eff->GetXaxis()->SetLabelSize(18);
  graph_mult_eff->GetXaxis()->SetTitleFont(43);
  graph_mult_eff->GetXaxis()->SetTitleSize(15);
  graph_mult_eff->GetXaxis()->SetTitleOffset(1.1);

  graph_mult_eff->GetYaxis()->SetTitle("#it{Efficiency}");
  graph_mult_eff->GetXaxis()->SetTitle("#it{Multiplicity}_{#it{sim}}");
  graph_mult_eff->Draw("AP");

  c8->SaveAs("NumSim_efficiency.png", "recreate");

  // Histogram for distance
  int num_bins_dist = 100; // Need to swap bins to log scale ones!!!!!!!!!!!!!!!!!
  double d_max = 5., d_min = 0.;
  TH1D *ver_dis = new TH1D("hist_distance", " ; #it{x}, cm ; ", num_bins_dist, d_min, d_max);
  ver_dis->Sumw2();

  for (unsigned int n = 0; n < mc_info.size(); ++n)
  {
    for (unsigned int i = 0; i < rec_info.size(); ++i)
    {
      if (rec_info[i].GetTopLabel() == (int)n)
      {
        // std::cout << n << "\t" << mc_info[n].EvId << "\t" <<  rec_info[i].GetTopLabel() << "\n";
        // std::cout << TMath::Sqrt( TMath::Power(mc_info[n].Vx - rec_info[i].Vx, 2) + TMath::Power(mc_info[n].Vy - rec_info[i].Vy, 2) +TMath::Power(mc_info[n].Vz - rec_info[i].Vz, 2) ) << "\n\n";
        ver_dis->Fill(TMath::Sqrt(TMath::Power(mc_info[n].Vx - rec_info[i].Vx, 2) + TMath::Power(mc_info[n].Vy - rec_info[i].Vy, 2) + TMath::Power(mc_info[n].Vz - rec_info[i].Vz, 2)));
      }
    }
  }

  TCanvas *c9 = new TCanvas;
  c9->cd();
  c9->SetGridx();
  c9->SetGridy();
  gPad->SetLogx();
  gPad->SetLogy();
  ver_dis->SetLineColor(kBlack);
  ver_dis->SetLineWidth(2);
  ver_dis->SetFillColorAlpha(15, 0.1);
  ver_dis->SetFillStyle(3244);

  ver_dis->GetYaxis()->SetLabelFont(43);
  ver_dis->GetYaxis()->SetLabelSize(26);
  ver_dis->GetYaxis()->SetTitleFont(43);
  ver_dis->GetYaxis()->SetTitleSize(20);
  ver_dis->GetYaxis()->SetTitleOffset(2.0);

  ver_dis->GetXaxis()->SetLabelFont(43);
  ver_dis->GetXaxis()->SetLabelSize(26);
  ver_dis->GetXaxis()->SetTitleFont(43);
  ver_dis->GetXaxis()->SetTitleSize(20);
  ver_dis->GetXaxis()->SetTitleOffset(1.1);

  ver_dis->Draw("hist");
  c9->SaveAs("distance_hist.png", "recreate");

  // Distance vs N_contr
  std::vector<double_t> plot_distance;
  std::vector<double_t> plot_NumContr;

  for (unsigned int n = 0; n < mc_info.size(); ++n)
  {
    for (unsigned int i = 0; i < rec_info.size(); ++i)
    {
      if (rec_info[i].GetTopLabel() == (int)n)
      {
        plot_NumContr.push_back(rec_info[i].NumContr);
        plot_distance.push_back(TMath::Sqrt(TMath::Power(mc_info[n].Vx - rec_info[i].Vx, 2) + TMath::Power(mc_info[n].Vy - rec_info[i].Vy, 2) + TMath::Power(mc_info[n].Vz - rec_info[i].Vz, 2)));
      }
    }
  }

  TGraph *graph_distance_vs_numcontr = new TGraph(plot_distance.size(), &plot_NumContr[0], &plot_distance[0]);

  TCanvas *c10 = new TCanvas;
  c10->cd();
  c10->SetGridx();
  c10->SetGridy();
  gPad->SetLogx();
  gPad->SetLogy();

  graph_distance_vs_numcontr->SetMarkerStyle(20);
  graph_distance_vs_numcontr->SetMarkerSize(1);
  graph_distance_vs_numcontr->SetMarkerColor(kBlue + 2);

  graph_distance_vs_numcontr->GetYaxis()->SetLabelFont(43);
  graph_distance_vs_numcontr->GetYaxis()->SetLabelSize(18);
  graph_distance_vs_numcontr->GetYaxis()->SetTitleFont(43);
  graph_distance_vs_numcontr->GetYaxis()->SetTitleSize(15);
  graph_distance_vs_numcontr->GetYaxis()->SetTitleOffset(2.0);

  graph_distance_vs_numcontr->GetXaxis()->SetLabelFont(43);
  graph_distance_vs_numcontr->GetXaxis()->SetLabelSize(18);
  graph_distance_vs_numcontr->GetXaxis()->SetTitleFont(43);
  graph_distance_vs_numcontr->GetXaxis()->SetTitleSize(15);
  graph_distance_vs_numcontr->GetXaxis()->SetTitleOffset(1.1);

  graph_distance_vs_numcontr->GetYaxis()->SetTitle("#it{distance}, cm");
  graph_distance_vs_numcontr->GetXaxis()->SetTitle("#it{N}_{#it{contr}}");

  // graph_distance_vs_numcontr->GetYaxis()->SetRangeUser(0.0001, 140. + 10.);

  graph_distance_vs_numcontr->Draw("AP");
  c10->SaveAs("Distance_vs_NumContr.png", "recreate");

  std::vector<double_t> hist_distance, hist_NumContr;

  for (unsigned int n = 0; n < mc_info.size(); ++n)
  {
    for (unsigned int i = 0; i < rec_info.size(); ++i)
    {
      if (rec_info[i].GetTopLabel() == (int)n)
      {
        hist_NumContr.push_back(rec_info[i].NumContr);
        hist_distance.push_back(TMath::Sqrt(TMath::Power(mc_info[n].Vx - rec_info[i].Vx, 2) + TMath::Power(mc_info[n].Vy - rec_info[i].Vy, 2) + TMath::Power(mc_info[n].Vz - rec_info[i].Vz, 2)));
        double d = TMath::Sqrt(TMath::Power(mc_info[n].Vx - rec_info[i].Vx, 2) + TMath::Power(mc_info[n].Vy - rec_info[i].Vy, 2) + TMath::Power(mc_info[n].Vz - rec_info[i].Vz, 2));
        ///if (d > 5){
        ///  std::cout << "distance : " << d << " cm \n" << "Sim vertex position : " << " ( " << mc_info[n].Vx << " , " << mc_info[n].Vy << " , " << mc_info[n].Vz << " ) \n"
        ///  << "Rec vertex position : " << " ( " << rec_info[i].Vx << " , " << rec_info[i].Vy << " , " << rec_info[i].Vz << " ) \n"
        ///  << "N_contr : " << rec_info[i].NumContr <<"\n";
        ///}
      }
    }
  }

  // log scale

  int nb_dnc = 50;
  double xbins_dnc[nb_dnc + 1], xmin_dnc = 2, xmax_dnc = *max_element(hist_NumContr.begin(), hist_NumContr.end());
  double ybins_dnc[nb_dnc + 1], ymin_dnc = 0.01, ymax_dnc = *max_element(hist_distance.begin(), hist_distance.end());
  double ax_dnc = std::log(xmax_dnc / xmin_dnc) / nb_dnc;
  double ay_dnc = std::log(ymax_dnc / ymin_dnc) / nb_dnc;
  for (int i = 0; i <= nb_dnc; i++)
  {
    // std::cout <<  xmin_dnc * std::exp(i * ax_dnc) << " ; " << ymin_dnc * std::exp(i * ay_dnc) <<std::endl;
    xbins_dnc[i] = xmin_dnc * std::exp(i * ax_dnc);
    ybins_dnc[i] = ymin_dnc * std::exp(i * ay_dnc);
  }
  // log scale
  TH2F *hist_distance_vs_numcontr = new TH2F("hist_distance_vs_numcontr", "", nb_dnc, xbins_dnc, nb_dnc, ybins_dnc);

  // log only for y axe
  //  TH2F *hist_distance_vs_numcontr = new TH2F("hist_distance_vs_numcontr", "",nb_dnc, 2, *max_element(hist_NumContr.begin(), hist_NumContr.end()), nb_dnc, ybins_dnc);

  // normal scale
  // TH2F *hist_distance_vs_numcontr = new TH2F("hist_distance_vs_numcontr", "", nb_dnc, 2., *max_element(hist_NumContr.begin(), hist_NumContr.end()) + 1, nb_dnc, 0., *max_element(hist_distance.begin(), hist_distance.end()) + 1);

  hist_distance_vs_numcontr->SetStats(0);

  for (unsigned int i = 0; i < hist_distance.size(); ++i)
  {
    hist_distance_vs_numcontr->Fill(hist_NumContr[i], hist_distance[i]);
    // std::cout << hist_distance[i] << "    " << hist_NumContr[i] << "\n";
  }

  TCanvas *c11 = new TCanvas;
  c11->cd();
  c11->SetGridx();
  c11->SetGridy();
  // c11->SetLogx();
  // c11->SetLogy();
  gPad->SetLogx();
  gPad->SetLogy();
  // gPad->SetLogz();

  gStyle->SetPalette(kRainBow);
  hist_distance_vs_numcontr->SetContour(1000);

  hist_distance_vs_numcontr->GetYaxis()->SetLabelFont(43);
  hist_distance_vs_numcontr->GetYaxis()->SetLabelSize(18);
  hist_distance_vs_numcontr->GetYaxis()->SetTitleFont(43);
  hist_distance_vs_numcontr->GetYaxis()->SetTitleSize(15);
  hist_distance_vs_numcontr->GetYaxis()->SetTitleOffset(2.0);

  hist_distance_vs_numcontr->GetXaxis()->SetLabelFont(43);
  hist_distance_vs_numcontr->GetXaxis()->SetLabelSize(18);
  hist_distance_vs_numcontr->GetXaxis()->SetTitleFont(43);
  hist_distance_vs_numcontr->GetXaxis()->SetTitleSize(15);
  hist_distance_vs_numcontr->GetXaxis()->SetTitleOffset(1.1);

  hist_distance_vs_numcontr->GetYaxis()->SetTitle("#it{distance}, cm");
  hist_distance_vs_numcontr->GetXaxis()->SetTitle("#it{N}_{#it{contr}}");
  // hist_distance_vs_numcontr->GetZaxis()->SetTitle("#it{Entries}");

  hist_distance_vs_numcontr->Draw("colz");
  c11->SaveAs("Hist_Distance_vs_NumContr.png", "recreate");*/
}