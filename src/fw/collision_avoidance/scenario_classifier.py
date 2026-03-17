

"""
Source: https://www.samgongustofa.is/media/log-og-reglur/COLREG-Consolidated-2018.pdf

1. Head-on scenario:

(a)     When two power-driven vessels are meeting on reciprocal or nearly reciprocal courses so as to
        involve risk of collision each shall alter her course to starboard so that each shall pass on the port
        side of the other.

(b)     Such a situation shall be deemed to exist when a vessel sees the other ahead or nearly ahead and
        by night she could see the masthead lights of the other in a line or nearly in a line and/or both
        sidelights and by day she observes the corresponding aspect of the other vessel.

(c)     When a vessel is in any doubt as to whether such a situation exists she shall assume that it does
        exist and act accordingly.

2. Overtaking scenario:

(a)     Notwithstanding anything contained in the Rules of Part B, Sections I and II any vessel overtaking
        any other shall keep out of the way of the vessel being overtaken.

(b)     A vessel shall be deemed to be overtaking when coming up with another vessel from a direction
        more than 22.5 degrees abaft her beam, that is, in such a position with reference to the vessel she is
        overtaking, that at night she would be able to see only the sternlight of that vessel but neither of her
        sidelights.

(c)     When a vessel is in any doubt as to whether she is overtaking another, she shall assume that this
        is the case and act accordingly.

(d)     Any subsequent alteration of the bearing between the two vessels shall not make the overtaking
        vessel a crossing vessel within the meaning of these Rules or relieve her of the duty of keeping clear
        of the overtaken vessel until she is finally past and clear.


"""
class ScenarioClassifier:
    def __init__(self):
        self.classify_range = 80 # radius at which ts triggers classification
        self.forget_range = self.classify_range + 10 # point at which ts loses classification


    def classification_active(self, dist_to_ts):
        """
        Acts as a trigger: when a target ship enters the classify range it is classified
        according to a scenario. Only if its distance is greater than the forget range
        the scenario is dropped (classification becomes inactive). This avoids
        rapid switching between scenarios
        """


    def classify_scenario(self, ts_heading, ts_pos, os_heading, os_pos, dist_to_ts):
        if self.classification_active(dist_to_ts):

